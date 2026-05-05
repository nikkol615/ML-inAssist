"""
 Модуль: ranking.py
 Назначение: Сервис ранжирования временных слотов (ML)
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import csv
import os
import logging
import pickle
import threading
from datetime import datetime
from typing import List, Optional

import numpy as np

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from app.schemas import CandidateSlot, UserContext

logger = logging.getLogger("ml_ranking")
logger.setLevel(logging.INFO)

DATA_DIR = "/app/data" if os.path.exists("/app/data") else "."
DATASET_FILE = os.path.join(DATA_DIR, "training_data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "ranking_model.pkl")
MIN_SAMPLES_TO_TRAIN = 20


class RankingService:
    """LightGBM-ранкер слотов с online learning из фидбека."""

    def __init__(self):
        self.model: Optional[lgb.LGBMClassifier] = None
        self.use_ml = False
        self._lock = threading.RLock()

        if not os.path.exists(DATASET_FILE):
            with open(DATASET_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["target", "hour", "day_of_week", "duration", "is_work", "hour_sin", "hour_cos"])

        if HAS_LIGHTGBM and os.path.exists(MODEL_FILE):
            try:
                with self._lock:
                    with open(MODEL_FILE, "rb") as f:
                        self.model = pickle.load(f)
                    self.use_ml = True
                logger.info("lgbm loaded")
            except Exception as e:
                logger.warning(f"model load err: {e}")

        if not self.use_ml:
            logger.info("using heuristics")

    def _extract_features(self, slot: CandidateSlot, context: UserContext) -> dict:
        """6 признаков: hour, weekday, duration, is_work, hour_sin/cos."""
        try:
            dt = datetime.fromisoformat(slot.start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(slot.end.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            dt = datetime.now()
            end_dt = dt

        duration_min = (end_dt - dt).total_seconds() / 60
        hour = dt.hour
        weekday = dt.weekday()

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        is_work = 1 if (context.work_start_hour <= hour < context.work_end_hour and weekday < 5) else 0

        return {
            "hour": hour,
            "day_of_week": weekday,
            "duration": duration_min,
            "is_work": is_work,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos
        }

    def _features_to_array(self, features: dict) -> np.ndarray:
        """dict → numpy array для LightGBM."""
        return np.array([
            features["hour"],
            features["day_of_week"],
            features["duration"],
            features["is_work"],
            features["hour_sin"],
            features["hour_cos"]
        ])

    def rank_slots(self, slots: List[CandidateSlot], context: UserContext) -> List[CandidateSlot]:
        """ML-ранжирование или rule-based fallback."""
        if not slots:
            return []

        scored = []

        for slot in slots:
            f = self._extract_features(slot, context)

            with self._lock:
                model = self.model
                use_ml = self.use_ml

            if use_ml and model:
                X = self._features_to_array(f).reshape(1, -1)
                try:
                    prob = model.predict_proba(X)[0][1]
                    score = prob * 1000
                except Exception as e:
                    logger.warning(f"predict err: {e}")
                    score = self._rule_based_score(f)
            else:
                score = self._rule_based_score(f)

            scored.append((slot, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def _rule_based_score(self, features: dict) -> float:
        """Fallback: приоритет рабочим часам, штраф выходным."""
        score = 0.0
        if features["is_work"]:
            score += 100
        if 10 <= features["hour"] <= 16:
            score += 20
        if features["day_of_week"] >= 5:
            score -= 50
        return score

    def save_training_data(self, context: UserContext, chosen: CandidateSlot, rejected: List[CandidateSlot]):
        """Online learning: сохраняет фидбек для переобучения."""
        data_rows = []

        f = self._extract_features(chosen, context)
        data_rows.append([1, f["hour"], f["day_of_week"], f["duration"], f["is_work"], f["hour_sin"], f["hour_cos"]])

        for slot in rejected:
            f = self._extract_features(slot, context)
            data_rows.append(
                [0, f["hour"], f["day_of_week"], f["duration"], f["is_work"], f["hour_sin"], f["hour_cos"]])

        try:
            with self._lock:
                with open(DATASET_FILE, "a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerows(data_rows)
                logger.info(f"+{len(data_rows)} samples")

                self._maybe_retrain_locked()

        except Exception as e:
            logger.error(f"save err: {e}")

    def _maybe_retrain(self):
        """Auto-retrain LightGBM при накоплении 20+ примеров."""
        with self._lock:
            self._maybe_retrain_locked()

    def _maybe_retrain_locked(self):
        """Auto-retrain LightGBM при накоплении 20+ примеров. Требует self._lock."""
        if not HAS_LIGHTGBM:
            logger.warning("no lgbm")
            return

        try:
            data = []
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)

            if len(data) < MIN_SAMPLES_TO_TRAIN:
                logger.info(f"need more data: {len(data)}/{MIN_SAMPLES_TO_TRAIN}")
                return

            X = np.array([
                [float(row["hour"]), float(row["day_of_week"]), float(row["duration"]),
                 float(row["is_work"]), float(row["hour_sin"]), float(row["hour_cos"])]
                for row in data
            ])
            y = np.array([int(row["target"]) for row in data])

            if len(np.unique(y)) < 2:
                logger.warning("need both classes")
                return

            class_counts = np.bincount(y)
            if np.min(class_counts) < 2:
                logger.warning("need at least two samples per class")
                return

            # stratify=y: сохраняем пропорцию классов 0/1 в train и test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=5,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            logger.info(f"trained, acc={accuracy:.1%}")

            tmp_model_file = MODEL_FILE + ".tmp"
            with open(tmp_model_file, "wb") as f:
                pickle.dump(model, f)
            os.replace(tmp_model_file, MODEL_FILE)

            self.model = model
            self.use_ml = True
            logger.info("model saved")

        except Exception as e:
            logger.error(f"train err: {e}")
