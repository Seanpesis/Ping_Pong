# Interactive Hand-Controlled Pong Game

An interactive Pong game built with Python, OpenCV, and MediaPipe Hands. Players control paddles using real-time hand movements captured by a webcam, with options to play against the computer or a friend. Features dynamic scoring, increasing ball speed, and a visually appealing gradient background. Demonstrates skills in computer vision, real-time processing, and Python programming.

## Features

- **Hand Tracking Control:** Use hand movements to control paddles in real-time via webcam.
- **Game Modes:** 
  - **Single-Player:** Play against an AI-controlled computer opponent.
  - **Multiplayer:** Play against a friend by tracking two hands simultaneously.
- **Dynamic Scoring:** Accurate scoring system that rewards the player who successfully sends the ball past the opponent.
- **Increasing Difficulty:** Ball speed increases with each hit to enhance the gameplay experience.
- **User Interface:** Includes a main menu, in-game HUD, and end-of-game screens with clear instructions.
- **Visuals:** Features a gradient background for better contrast and visual appeal.

## Technologies Used

- **Python:** Core programming language for game logic.
- **OpenCV:** Handles video capture and image processing.
- **MediaPipe Hands:** Facilitates real-time hand tracking and gesture recognition.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/interactive-pong.git
   cd interactive-pong
Create and Activate a Virtual Environment (Optional but Recommended):

bash
Copy code
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the Required Packages:

bash
Copy code
pip install -r requirements.txt
Run the Game:

bash
Copy code
python pong.py
Usage
Start Screen:
Press 'C' to play against the computer.
Press 'F' to play against a friend.
Game Controls:
Use your hand movements in front of the webcam to move your paddle up and down.
Scoring:
The first player to reach 3 points wins the game.
After the game ends, press:
'Q' to quit.
'R' to restart the game.
'M' to return to the main menu.
Screenshots
Add screenshots of your game here to showcase the interface and gameplay.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or feature requests.

License
This project is licensed under the MIT License.

Acknowledgements
Inspired by classic Pong game mechanics.
Thanks to the numerous tutorials and resources available online that guided the development of this project.
Special thanks to MediaPipe for their excellent hand tracking solutions.
javascript
Copy code





