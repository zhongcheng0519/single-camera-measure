import cv2
import numpy as np


W_IN_CM = 21
H_IN_CM = 16.5
DPI = 300
GRID_COL = 14
GRID_ROW = 11


if __name__ == '__main__':
    w_cm = W_IN_CM
    h_cm = H_IN_CM
    dpi = DPI
    grid_col = GRID_COL
    grid_row = GRID_ROW
    cm_per_inch = 2.54
    dpcm = dpi/cm_per_inch
    cols = w = int(w_cm * dpcm)
    rows = h = int(h_cm * dpcm)
    cm_per_grid_col = w_cm/grid_col
    cm_per_grid_row = h_cm/grid_row

    chessboard = np.zeros((rows, cols), np.uint8)
    row_seq0 = ((-1)**np.arange(0, grid_col)+1)/2*255
    row_seq1 = ((-1)**(np.arange(0, grid_col)+1)+1)/2*255
    pattern = np.zeros((grid_row, grid_col), np.float64)
    for gr in range(grid_row):
        pattern[gr, :] = row_seq0 if gr % 2 == 0 else row_seq1

    for r in range(rows):
        for c in range(cols):
            grid_i = int(r/dpcm/cm_per_grid_row)
            grid_j = int(c/dpcm/cm_per_grid_col)
            chessboard[r, c] = pattern[grid_i, grid_j]
    cv2.imshow("win2", chessboard)
    cv2.waitKey(0)
    cv2.imwrite('chessboard_300.bmp', chessboard)
    cv2.destroyAllWindows()
