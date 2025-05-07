import sys

from dotenv import load_dotenv
from hand_gesture_regonition.home_assistant_commands import HomeAssistantCommands
from hand_gesture_regonition.program import Program
from hand_gesture_regonition.trainer import main as trainer_main


def main():
    load_dotenv()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'trainer':
        trainer_main()
    else:
        Program.get().run()