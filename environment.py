import numpy as np
from math import fabs, log2
from tkinter import Frame, Label, CENTER

import logic
import constants as c
from agent import Agent
from constants import args


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
                    done = True
            return self.matrix, score, done, win, move_complete, tiles_moved

    def reward(self, score):
        reward = log2(score) if score > 0 else 0
        temp_matrix = np.copy(self.matrix)

        # TODO: Make penalty percentage based on final reward
        max_index = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
        reward -= np.min([
            fabs(max_index[0] - 0) + fabs(max_index[1] - 0),
        ]) / 6  # penalize distance from corner

        # TODO: Make penalty percentage based on final reward
        dist = 0
        while np.any(temp_matrix):
            temp_matrix[max_index] = 0
            max_index_ = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            dist += fabs(max_index[0] - max_index_[0]) + fabs(max_index[1] - max_index_[1])
            max_index = max_index_
        reward -= dist / 96  # penalize distance to closest value

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
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym

        if key == c.KEY_DQN:
            agent = Agent()

            agent.load_state(args.filepath)

            state, end, win = self.get_state()

            while not end:
                action = agent.choose_action(state, 1, 1, False)

                print(action)

                if action == 0:
                    self.matrix, done, score, _ = logic.up(self.matrix)
                elif action == 1:
                    self.matrix, done, score, _ = logic.down(self.matrix)
                elif action == 2:
                    self.matrix, done, score, _ = logic.left(self.matrix)
                else:
                    self.matrix, done, score, _ = logic.right(self.matrix)

                if done:
                    self.game_done()

                state, end, win = self.get_state()
        else:
            if key == c.KEY_QUIT:
                exit()
            if key == c.KEY_BACK and len(self.history_matrices) > 1:
                self.matrix = self.history_matrices.pop()
                self.update_grid_cells()
            elif key in self.commands:
                self.matrix, done, score = self.commands[key](self.matrix)
                if done:
                    self.game_done()

    def game_done(self):
        self.matrix = logic.add_two(self.matrix)
        self.history_matrices.append(self.matrix)
        self.update_grid_cells()
        if logic.game_state(self.matrix) == 'win':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        if logic.game_state(self.matrix) == 'lose':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
