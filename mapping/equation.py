import math

class equation:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.Vflag = False
        if x1 == x2 :
            self.slope = 99999999999999999999
            self.Vflag = True
        else:
            self.slope = (y2-y1) / (x2-x1)

        if self.slope == 0 :
            self.theta = - math.pi / 2
        else:
            self.theta = math.atan(-1 / self.slope)
        self.rho = math.sin(self.theta) * (self.y1 - self.slope * self.x1)
    
    def length(self):
        return math.sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2)

    def distance_from(self, line):
        if self.Vflag or line.Vflag : 
            return abs(self.x1 - line.x1)
        
        ## Approch 1
        cx1 = (self.x1 + self.x2) / 2
        cy1 = (self.y1 + self.y2) / 2
        cx2 = (line.x1 + line.x2) / 2
        cy2 = (line.y1 + line.y2) / 2
        a1 =  math.sqrt((cy1 - cy2) ** 2 + (cx1 - cx2) ** 2)
        
        ## Approch 2
        m = (self.slope + line.slope) / 2
        b1 = self.y1 - m * self.x1
        b2 = line.y1 - m * line.x1
        a2 = abs(b1 - b2) / math.sqrt(m ** 2 + 1)
        return min(a1, a2)
        
    def y_equal(self, x):
        return self.slope * (x - self.x1) + self.y1


    def x_equal(self, y):
        if self.Vflag : return self.x1
        return (y - self.y1) / self.slope + self.x1

    def is_vertical(self):
        if self.Vflag  or self.slope > 1 or self.slope < -1: 
            return True
        return False


    def info(self):
        return self.x1, self.y1, self.x2, self.y2

