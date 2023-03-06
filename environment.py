import numpy as np
from math import fabs, log2
from tkinter import Frame, Label, CENTER

import time
import utils
import logic
import config as c
from agent import Agent
from config import args


class GameEngine:
    def __init__(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.score = 0
        self.moves = {
            0: logic.up,
            1: logic.down,
            2: logic.left,
            3: logic.right,
        }

    def reset(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.score = 0
        return self.matrix

    def step(self, move):
        done = False
        win = False
        if move in self.moves:
            self.matrix, move_complete, score, tiles_moved = self.moves[move](self.matrix)
            self.score += score
            if move_complete:
                self.matrix = logic.add_two(self.matrix)
                if logic.game_state(self.matrix) == 'win':
                    done = True
                    win = True
                if logic.game_state(self.matrix) == 'lose':
                    if np.max(self.matrix) >= 2048:
                        win = True
                    done = True

            return self.matrix, score, done, win, move_complete, tiles_moved

    def reward(self, score):
        reward = log2(score) if score > 0 else 0
        temp_matrix = np.copy(self.matrix)

        max_index = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
        penalty = min(np.min([
            fabs(max_index[0] - 0) + fabs(max_index[1] - 0),
        ]) / 6, 1)  # Penalize distance from corner

        reward -= reward * penalty * args.penalty  # Penalty affects % of the reward

        dist = 0
        while np.any(temp_matrix):
            temp_matrix[max_index] = 0
            max_index_ = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            dist += fabs(max_index[0] - max_index_[0]) + fabs(max_index[1] - max_index_[1])
            max_index = max_index_
        penalty = min(dist / 96, 1)  # Penalize distance to closest value

        reward -= reward * penalty * args.penalty  # Penalty affects % of the reward

        return reward


class GameRender(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,

            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,

            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrices = []
        self.update_grid_cells()

        self.agent = Agent()
        self.agent.load_complete_state(args.complete_filepath)

        self.mainloop()

    def get_state(self):
        state = logic.game_state(self.matrix)
        done = False
        win = False
        if state == 'win':
            done = True
            win = True
        elif state == 'lose':
            done = True

        return self.matrix, done, win

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update()

    def key_down(self, event):
        key = event.keysym

        if key == c.KEY_DQN:

            observation, end, win = self.get_state()

            while not end:
                state = utils.transform_state(np.array(observation))
                action = self.agent.choose_complete_action(state)

                if action == 0:
                    self.matrix, done, score, _ = logic.up(self.matrix)
                elif action == 1:
                    self.matrix, done, score, _ = logic.down(self.matrix)
                elif action == 2:
                    self.matrix, done, score, _ = logic.left(self.matrix)
                else:
                    self.matrix, done, score, _ = logic.right(self.matrix)

                if done:
                    self.after(100, self.game_done())

                observation, end, win = self.get_state()
        else:
            if key == c.KEY_QUIT:
                exit()
            if key == c.KEY_BACK and len(self.history_matrices) > 1:
                self.matrix = self.history_matrices.pop()
                self.update_grid_cells()
            elif key in self.commands:
                self.matrix, done, score, _ = self.commands[key](self.matrix)
                if done:
                    self.game_done()

    def game_done(self):
        self.matrix = logic.add_two(self.matrix)
        self.update_grid_cells()
        if logic.game_state(self.matrix) == 'win':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        if logic.game_state(self.matrix) == 'lose':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
