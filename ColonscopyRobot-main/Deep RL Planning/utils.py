import numpy as np
from math import floor

def bresenham(x1, y1, x2, y2):
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)

        dx = abs(x2-x1)
        dy = abs(y2-y1)

        steep = dy > dx
        if steep:
            t = dx
            dx = dy
            dy = t

        if dy == 0:
            q = np.zeros([int(dx+1), 1])
        else:
            q = [a for a in np.arange(floor(dx/2), -dy * dx + floor(dx/2) - dy, -dy)]
            q = np.hstack([0, (np.diff(np.mod(q, dx)) >= 0).astype(float)])

        if steep:
            if y1 <= y2:
                y = np.arange(y1, y2+1)
            else:
                y = np.arange(y1, y2-1, -1)
            if x1 <= x2:
                x = x1 + np.cumsum(q)
            else:
                x = x1 - np.cumsum(q)
        else:
            if x1 <= x2:
                x = np.arange(x1, x2+1)
            else:
                x = np.arange(x1, x2-1, -1)
            if y1 <= y2:
                y = y1 + np.cumsum(q)
            else:
                y = y1 - np.cumsum(q)

        if len(x) is not len(y):
            print("bresenham algorithm issues, return list not same!")
        

        return x, y     
