from graph import build_graph
from datetime import datetime
from statistics_tracker import StatisticsTracker
import json
import os

graph = build_graph()
log_file = "A:/intership/task 2/logs/langgraph_interactions.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)


def log(state):
    """Enhanced logging with proper fallback tracking"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": state.get("original_input", state["input"]),
        "initial_labels": state.get("initial_labels", []),
        "initial_conf": state.get("max_conf", None),
        "fallback_triggered": False,  # Will be updated below
        "final_labels": state.get("final_labels", state.get("initial_labels", []))
    }

    # Check if any fallback was actually used
    fallback_used = False

    # Enhanced logging for bonus features
    if state.get("backup_model_used"):
        log_entry["backup_model_used"] = True
        log_entry["backup_labels"] = state.get("backup_labels", [])
        log_entry["backup_conf"] = state.get("backup_conf", 0)

    # Check for user clarification
    if state.get("clarified"):
        log_entry["fallback_method"] = "user"
        log_entry["clarified_input"] = state.get("clarified_input", "")
        fallback_used = True

    # Check for backup model usage
    elif state.get("fallback_method") == "backup_model":
        log_entry["fallback_method"] = "backup_model"
        fallback_used = True

    # Update fallback_triggered based on actual usage
    log_entry["fallback_triggered"] = fallback_used

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def print_help():
    print("\n🔮 LangGraph Emotion Classifier - Commands:")
    print("• Type any text to classify emotions")
    print("• 'stats' - Show performance statistics")
    print("• 'charts' - Generate visualization charts")
    print("• 'help' - Show this help message")
    print("• 'exit' - Quit the application\n")


def handle_special_commands(user_input):
    """Handle special CLI commands"""
    if user_input.lower() == "stats":
        tracker = StatisticsTracker(log_file)
        tracker.load_logs()
        tracker.print_cli_stats()
        return True
    elif user_input.lower() == "charts":
        tracker = StatisticsTracker(log_file)
        tracker.generate_all_visualizations()
        return True
    elif user_input.lower() == "help":
        print_help()
        return True
    return False


print("🔮 LangGraph Emotion Classifier with Self-Healing & Analytics")
print_help()

while True:
    user_input = input("📝 Your input > ").strip()
    if user_input.lower() == "exit":
        break

    # Handle special commands
    if handle_special_commands(user_input):
        continue

    if not user_input:
        continue

    state = {"input": user_input, "original_input": user_input}
    result = graph.invoke(state, config={"recursion_limit": 5})

    # Check if backup model provided better results
    backup_used_as_primary = (result.get("fallback_method") == "backup_model")

    # Display backup model results if consulted but not used as primary
    if result.get("backup_model_used") and not backup_used_as_primary:
        print(f"\n🔄 Backup model consulted:")
        print(f"Backup labels: {result.get('backup_labels', [])}")
        print(f"Backup confidence: {result.get('backup_conf', 0):.2f}")

    # Handle user clarification fallback
    if result.get("fallback_needed") and not result.get("clarified"):
        if not backup_used_as_primary:
            print(f"\n⚠️  Low confidence detected from both models.")
            print(f"🤔 Please help clarify your intent.")
            print(f"Primary model confidence: {result['max_conf']:.2f}")
            if result.get("backup_conf"):
                print(f"Backup model confidence: {result['backup_conf']:.2f}")

            clarification = input("Describe the emotion you intended (or type 'neutral'): ").strip()

            # Process the clarification
            clarified_state = {
                "input": clarification,
                "original_input": user_input,  # Keep original input
                "clarified": True,
                "fallback_method": "user"
            }

            clarified_result = graph.invoke(clarified_state, config={"recursion_limit": 5})

            # Properly merge results
            result.update({
                "clarified": True,
                "fallback_method": "user",
                "fallback_needed": False,
                "final_labels": clarified_result.get("final_labels", clarified_result.get("initial_labels", [])),
                "clarified_input": clarification,
                # Keep original confidence and backup info
                "max_conf": result["max_conf"],  # Keep original confidence
                "backup_model_used": result.get("backup_model_used", False),
                "backup_labels": result.get("backup_labels", []),
                "backup_conf": result.get("backup_conf", 0)
            })

    # Display results
    print("\n📊 Final Prediction:")
    print(f"Text        : {result['original_input']}")
    print(f"Labels      : {result.get('final_labels', result.get('initial_labels', []))}")
    print(f"Confidence  : {result.get('max_conf', 0):.2f}")

    # Enhanced fallback reporting
    if result.get('clarified'):
        print(f"Fallback    : Used user clarification")
        if result.get('clarified_input'):
            print(f"Clarification: '{result.get('clarified_input', '')}'")
    elif result.get('fallback_method') == 'backup_model':
        print(f"Fallback    : Used backup model (higher confidence)")
    else:
        print(f"Fallback    : Not triggered")

    # Show backup model info if used
    if result.get("backup_model_used") and result.get("backup_labels"):
        backup_status = "→ USED AS PRIMARY" if backup_used_as_primary else "→ consulted only"
        print(f"Backup Model: {result['backup_labels']} (conf: {result.get('backup_conf', 0):.2f}) {backup_status}")

    print("-" * 50)
    log(result)

print("\n👋 Thanks for using the Emotion Classifier!")
print("💡 Run 'python statistics_tracker.py' to view detailed analytics.")