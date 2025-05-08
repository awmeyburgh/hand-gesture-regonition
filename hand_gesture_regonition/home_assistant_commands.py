import os
from homeassistant_api import Client
from hand_gesture_regonition.gesture_commands import GestureCommands


class HomeAssistantCommands(GestureCommands):
    def __init__(self):
        super().__init__()
        
        self.client = Client(
            os.environ['HOME_ASSISTANT_URL'],
            os.environ['HOME_ASSISTANT_TOKEN'],
        )
        
    def turn_on(self):
        self.client.trigger_service('switch', 'turn_on', entity_id="switch.studyfanswitch_study_light")
        
    def turn_off(self):
        self.client.trigger_service('switch', 'turn_off', entity_id="switch.studyfanswitch_study_light")

    def call(self, key):
        if key == 's_right_thumb_up':
            self.turn_on()
        elif key == 's_right_thumb_down':
            self.turn_off()
        
    def close(self):
        self.client.cache_session.close()