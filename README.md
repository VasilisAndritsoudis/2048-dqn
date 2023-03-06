<a id="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
![GitHub last commit][last-commit-shield]
![GitHub top language][top-language-shield]
![Total Lines][lines-shield]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/VasilisAndritsoudis/2048-dqn">
    <img src="img/dqn_2048.png" alt="Logo" width="750" height="150">
  </a>

<h3 align="center">2048 Deep Q-Learning Network Agent</h3>

  <p align="center">
    Teaching a neural network agent to play the game 2048
    <br />
    <a href="https://github.com/VasilisAndritsoudis/2048-dqn"><strong>Explore the docs »</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project <a id="about-the-project"></a>

This project is an attempt at creating an agent, using the capabilities of Neural Networks, 
who is then capable of beating the widely known game [2048](https://play2048.co). The project has 
been designed based on the Double Deep Q-Learning networks and can support a variety of different 
network models (e.g. linear, convolutional), as well as different state encoding functions 
(e.g. as-is, log2, one-hot). Every other possible hyperparameter is also easily tunable. Last but not 
least an attempt was made to construct a custom epsilon decay strategy which also can be activated or 
deactivated. All the mentioned parameters can be easily tuned in the `config.py` file.

Along with the Neural Networks and the agents, a game UI was made which can be attached with
a trained model in order to try it out in a real hands-on scenario. The game can be launched
by running the `main.py` file and by pressing `W` (winner winner chicken dinner) the specified
agent from the configuration is activated and will beat the game.

Only the best performing network has been uploaded. If you wish to try 
out different network configurations, the networks have to be trained locally. A system with 
the following specifications (using Cuda):
* CPU: _AMD Ryzen 5 5600X_
* RAM: _16GB @ 3600Hz_
* GPU: _NVIDIA GeForce RTX 3060 Ti_

It takes about 6000 epochs (30 minutes) for the Network to start winning, and it takes about
15000 epochs (2 hours) for the Network to reach 10% win rate. The state of the Network is saved
at points throughout the training (can be specified in the `config.py` file) so it can be
trained in more than one session.

All the trainings and tests have been performed on the vanilla game, meaning each game ends when
the 2048 tile is reached. It is also possible to change the end tile in order to test the full
capabilities of the agents and push them to their limits. All the tested configurations have been
saved in the `Training.xlsx` file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- BUILT WITH -->
### Built With <a id="built-with"></a>

![Python][python-shield] 
![PyTorch][pytorch-shield] 
![NumPy][numpy-shield]
![nVIDIA][nvidia-shield]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started <a id="getting-started"></a>
<a id="getting-started"></a>

This project is built using `pytorch` and the use of `cuda` is strongly recommended.

We will use `pip` as package manager, PyTorch Build `1.12.1` and `CUDA 11.3`. You are free to change these
versions to your liking.

To get a local copy up and running follow these simple instruction steps.

### Prerequisites <a id="prerequisites"></a>

* [PyTorch][pytorch-url]
  ```sh
  $ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  ```
* [Tensorboard][tensorboard-url]
  ```sh
  $ pip install tensorboard
  ```
 
### Installation & Run <a id="installation"></a>
1. Clone the repo
   ```sh
   $ git clone https://github.com/VasilisAndritsoudis/2048-dqn.git
   ```
2. Train an agent
   1. Start the training
      ```sh
      $ python train.py
      ```
      Training hyperparameters can be found in `config.py`, or you can set them using
      ```sh
      $ python train.py -h
      ```
   2. Monitor runtime statistics with `tensorboard`
      1. Start the tensorboard service, `--logdir` can be a specific train scenario (eg. `runs/conv2d/one-hot`)
         ```sh
         $ tensorboard --logdir='runs' --load_fast=false
         ```
      2. Open the tensorboard service from your browser
         ```url
         http://localhost:6006/
         ```
3. Run the game (when at least one agent is trained)
   ```sh
   $ python main.py
   ```
   To activate the agent press `W`

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- DEMO EXAMPLES -->
## Demo <a id="demo"></a>
Using the configuration of the best model, a real-time run of the game is shown below. Keep in mind
the best model reaches ~15% win rate.

<div align="center">
  <a href="https://github.com/VasilisAndritsoudis/2048-dqn">
    <img src="img/game_animation.gif" alt="Logo">
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Future Ideas <a id="roadmap"></a>

* [ ] Train and test games further than tile 2048
* [ ] Implement move suggestions based on the agent predictions
* [ ] Create an API to publish the trained agent.
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
<!-- ## Contributing <a id="contributing"></a>

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact <a id="contact"></a>

[![GitHub Profile][github-shield]][github-url]
[![Twitter Profile][twitter-shield]][twitter-url]
[![LinkedIn Profile][linkedin-shield]][linkedin-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments <a id="acknowledgments"></a>
* [2048](https://play2048.co)
* [Learning 2048 with Deep Reinforcement Learning](https://cs.uwaterloo.ca/~mli/zalevine-dqn-2048.pdf)
* [A puzzle for AI](https://towardsdatascience.com/a-puzzle-for-ai-eb7a3cb8e599)
* [Best README Template](https://github.com/othneildrew/Best-README-Template)
* [Markdown Badges](https://github.com/Ileriayo/markdown-badges)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/VasilisAndritsoudis/2048-dqn.svg?style=for-the-badge
[contributors-url]: https://github.com/VasilisAndritsoudis/2048-dqn/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/VasilisAndritsoudis/2048-dqn.svg?style=for-the-badge
[stars-url]: https://github.com/VasilisAndritsoudis/2048-dqn/stargazers

[lines-shield]: https://img.shields.io/tokei/lines/github/VasilisAndritsoudis/2048-dqn?style=for-the-badge
[last-commit-shield]: https://img.shields.io/github/last-commit/VasilisAndritsoudis/2048-dqn?style=for-the-badge
[top-language-shield]: https://img.shields.io/github/languages/top/VasilisAndritsoudis/2048-dqn?style=for-the-badge

[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[pytorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[nvidia-shield]: https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white

[pytorch-url]: https://pytorch.org
[tensorboard-url]: https://www.tensorflow.org/tensorboard

[github-shield]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/VasilisAndritsoudis
[twitter-shield]: https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white
[twitter-url]: https://twitter.com/V_Andri_
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/vasilis-andritsoudis-b1b552217
