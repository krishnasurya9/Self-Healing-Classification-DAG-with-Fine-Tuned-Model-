import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
import os


class StatisticsTracker:
    def __init__(self, log_file="A:/intership/task 2/logs/langgraph_interactions.log"):
        self.log_file = log_file
        self.stats = {
            "total_predictions": 0,
            "fallback_triggered": 0,
            "backup_model_used": 0,
            "user_clarification_used": 0,
            "confidence_scores": [],
            "fallback_methods": defaultdict(int),
            "timestamps": []
        }

    def load_logs(self):
        """Load and parse log file"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.process_entry(entry)
                except json.JSONDecodeError:
                    continue

    def process_entry(self, entry):
        """Process a single log entry"""
        self.stats["total_predictions"] += 1

        # Track confidence scores
        if entry.get("initial_conf"):
            self.stats["confidence_scores"].append(entry["initial_conf"])

        # Track fallback usage
        if entry.get("fallback_triggered"):
            self.stats["fallback_triggered"] += 1
            method = entry.get("fallback_method", "unknown")
            self.stats["fallback_methods"][method] += 1

            if method == "backup_model":
                self.stats["backup_model_used"] += 1
            elif method == "user":
                self.stats["user_clarification_used"] += 1

        # Track timestamps
        if entry.get("timestamp"):
            self.stats["timestamps"].append(entry["timestamp"])

    def generate_confidence_curve(self, save_path="confidence_curve.png"):
        """Generate confidence curve visualization"""
        if not self.stats["confidence_scores"]:
            print("No confidence data available")
            return

        plt.figure(figsize=(12, 6))

        # Plot 1: Confidence scores over time
        plt.subplot(1, 2, 1)
        scores = self.stats["confidence_scores"]
        plt.plot(range(len(scores)), scores, 'b-o', alpha=0.7, markersize=4)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Confidence Threshold')
        plt.xlabel('Prediction Number')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Scores Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Confidence distribution
        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_fallback_statistics(self, save_path="fallback_stats.png"):
        """Generate fallback statistics visualization"""
        if self.stats["total_predictions"] == 0:
            print("No prediction data available")
            return

        plt.figure(figsize=(15, 5))

        # Plot 1: Fallback frequency pie chart
        plt.subplot(1, 3, 1)
        no_fallback = self.stats["total_predictions"] - self.stats["fallback_triggered"]
        sizes = [no_fallback, self.stats["fallback_triggered"]]
        labels = ['No Fallback', 'Fallback Used']
        colors = ['lightgreen', 'lightcoral']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Fallback Usage Distribution')

        # Plot 2: Fallback methods
        plt.subplot(1, 3, 2)
        if self.stats["fallback_methods"]:
            methods = list(self.stats["fallback_methods"].keys())
            counts = list(self.stats["fallback_methods"].values())
            plt.bar(methods, counts, color=['orange', 'purple', 'cyan'][:len(methods)])
            plt.xlabel('Fallback Method')
            plt.ylabel('Count')
            plt.title('Fallback Methods Used')
            plt.xticks(rotation=45)

        # Plot 3: Performance metrics
        plt.subplot(1, 3, 3)
        metrics = ['Total\nPredictions', 'Fallback\nTriggered', 'Backup Model\nUsed', 'User\nClarification']
        values = [
            self.stats["total_predictions"],
            self.stats["fallback_triggered"],
            self.stats["backup_model_used"],
            self.stats["user_clarification_used"]
        ]

        bars = plt.bar(metrics, values, color=['blue', 'red', 'green', 'orange'])
        plt.ylabel('Count')
        plt.title('System Performance Metrics')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(value), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_cli_stats(self):
        """Print statistics in CLI format"""
        print("\n" + "=" * 50)
        print("📊 SYSTEM PERFORMANCE STATISTICS")
        print("=" * 50)

        if self.stats["total_predictions"] == 0:
            print("No predictions recorded yet.")
            return

        fallback_rate = (self.stats["fallback_triggered"] / self.stats["total_predictions"]) * 100
        avg_confidence = np.mean(self.stats["confidence_scores"]) if self.stats["confidence_scores"] else 0

        print(f"🔢 Total Predictions: {self.stats['total_predictions']}")
        print(f"⚠️  Fallback Rate: {fallback_rate:.1f}%")
        print(f"📈 Average Confidence: {avg_confidence:.3f}")
        print(f"🤖 Backup Model Used: {self.stats['backup_model_used']} times")
        print(f"👤 User Clarification: {self.stats['user_clarification_used']} times")

        if self.stats["confidence_scores"]:
            print(
                f"📊 Confidence Range: {min(self.stats['confidence_scores']):.3f} - {max(self.stats['confidence_scores']):.3f}")

        print("\n📋 Fallback Methods:")
        for method, count in self.stats["fallback_methods"].items():
            print(f"   • {method}: {count} times")

        print("=" * 50)

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        self.load_logs()
        print("📊 Generating confidence curve...")
        self.generate_confidence_curve()
        print("📊 Generating fallback statistics...")
        self.generate_fallback_statistics()
        self.print_cli_stats()


# CLI command for statistics
if __name__ == "__main__":
    tracker = StatisticsTracker()
    tracker.generate_all_visualizations()