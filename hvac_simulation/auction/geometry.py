""" This Module implements classes for 2D geometry objects used for clearing """

import matplotlib.pyplot as plt

class Point:
    """
    The Point class implements a 2D point

    Attributes:
        x (float) - the x-coordinate
        y (float) - the y-coordinate
    Methods:
        __init__ - constructor
        __repr__ - string representation
        __eq__ - equality operator
    """
    def __init__(self,x,y=None):
        if isinstance(x, tuple):
            self.x, self.y = float(x[0]), float(x[1])
        else:
            self.x = float(x)
            self.y = float(y)

    def __repr__(self):
        return f"Point({self.x},{self.y})"

    def __eq__(self,other):
        if type(other) in (tuple,list):
            other = Point(*other)
        if self.x is None or self.y is None:
            if hasattr(other,"x") and other.x is None:
                return True
            if hasattr(other,"y") and other.y is None:
                return True
            return other is None 
        return hasattr(other,"x") and self.x == other.x and hasattr(other,"y") and self.y == other.y

class Line:
    """
    The Line class implements a 2D line segment

    Attributes:
        A (Point) - the start point
        B (Point) - the end point
    Methods:
        __init__ - constructor
        __repr__ - string representation
        __eq__ - equality operator
        isin - determine if a Point object instance is on the line
        intersect - calculate the intersection with another line
        reduce_point - find a representative point on the line segment based on xcollapse, ycollapse, xycollapse functions
    """

    def __init__(self, A, B):
        self.A = A if isinstance(A, Point) else Point(A)
        self.B = B if isinstance(B, Point) else Point(B)

    def __repr__(self):
        return f"Line(({self.A.x},{self.A.y}),({self.B.x},{self.B.y}))"

    def __eq__(self,other):
        if type(other) in (tuple,list):
            other = Line(*other)
        return ( self.A == other.A and self.B == other.B ) \
            or (self.A == other.B and self.B == other.A )

    def isin(self, other):
        """Determine whether a point is on a line

        Arguments:
        - other (Point): point to evaluate

        Returns:
        - bool: True if point is on line else False
        """
        if isinstance(other, list):
            other = Point(*other)
        dx, dy = (self.A.x-self.B.x), (self.A.y-self.B.y)
        if dx == 0:
            return min(self.A.y,self.B.y) <= other.y <= max(self.A.y,self.B.y)
        if dy == 0:
            return min(self.A.x,self.B.x) <= other.x <= max(self.A.x,self.B.x)
        return (other.x-self.B.x)/dx == (other.y-self.B.y)/dy
    
    def intersect(self, other):
        """Calculate the intersection with another line

        Arguments:
        - other (Line): line to intersect

        Returns:
        - Point/Line: intersection point or line

        """
        x1, y1 = self.A.x, self.A.y
        x2, y2 = self.B.x, self.B.y
        x3, y3 = other.A.x, other.A.y
        x4, y4 = other.B.x, other.B.y
        
        if any([
                max(x1,x2) < min(x3,x4),
                max(x3,x4) < min(x1,x2),
                max(y1,y2) < min(y3,y4),
                max(y3,y4) < min(y1,y2),
            ]): # no overlap
                return None

        m1 = (y2-y1)/(x2-x1) if x1 != x2 else None
        m2 = (y4-y3)/(x4-x3) if x3 != x4 else None

        if m1 is None: # L1 is vertical
            c1 = None
            if m2 is None: # L2 is vertical
                c2 = None
                if x1 == x3: # L1 and L2 may overlap
                    yy1, yy2 = min(y1,y2), max(y1,y2)
                    yy3, yy4 = min(y3,y4), max(y3,y4)
                    x0 = x1
                    y0 = [min(yy2,yy4), max(yy1,yy3)] if min(yy2,yy4) != max(yy1,yy3) else max(yy1, yy3) # overlap range
            else: # L2 is sloped
                c2 = y3 - m2*x3 # y-intercept
                x0 = x1
                y0 = c2 + m2*x1
        elif m2 is None: # L2 vertical (L1 is not vertical)
            c1 = y1-m1*x1 # y-intercept
            c2 = None
            x0 = x3
            y0 = m1*x3+c1
        elif m1 == 0.0: # L1 horizontal
            c1 = y1
            if m2 == 0.0: # L2 horizontal
                c2 = y3
                if y1 == y3: # L1 and L2 may overlap
                    xx1,xx2 = min(x1,x2),max(x1,x2) # reorder points
                    xx3,xx4 = min(x3,x4),max(x3,x4)
                    x0 = [min(xx2,xx4), max(xx1,xx3)] if min(xx2,xx4) != max(xx1,xx3) else max(xx1,xx3) # overlap range
                    y0 = y1
            else: # L2 is sloped
                c2 = y3 - m2*x3
                x0 = (c1-c2)/m2
                y0 = y1
        elif m2 == 0.0: # L2 horizontal (L1 is not horizontal)
            c1 = y1 - m1*x1
            c2 = y3
            x0 = (c2-c1)/m1 if m1 != 0.0 else x1
            y0 = c2
        else: # L1 and L2 are both sloped
            c1 = y1 - m1*x1
            c2 = y3 - m2*x3
            if m1 == m2: # L1 and L2 are parallel
                if c1 == c2: # L1 and L2 may overlap
                    xx1,xx2 = min(x1,x2),max(x1,x2) # reorder points
                    xx3,xx4 = min(x3,x4),max(x3,x4)
                    yy1,yy2 = min(y1,y2),max(y1,y2)
                    yy3,yy4 = min(y3,y4),max(y3,y4)
                    x0 = [min(xx2,xx4), max(xx1,xx3)] if min(xx2,xx4) != max(xx1,xx3) else max(xx1,xx3) # overlap range
                    y0 = [min(yy2,yy4), max(yy1,yy3)] if min(yy2,yy4) != max(yy1,yy3) else max(yy1,yy3) # overlap range
            else:
                x0 = (c1-c2)/(m2-m1)
                y0 = m1*x0+c1
        if x0 is None or y0 is None:
            xy = None
        elif isinstance(x0, list):
            if isinstance(y0, list):
                xy = Line(Point(x0[0], y0[0]), Point(x0[1], y0[1]))
            else:
                xy = Line(Point(x0[0], y0), Point(x0[1], y0))
        elif isinstance(y0, list):
            xy = Line(Point(x0, y0[0]), Point(x0, y0[1]))
        elif min(x1,x2) <= x0 <= max(x1,x2) and min(y1,y2) <= y0 <= max(y1,y2) \
                and min(x3,x4) <= x0 <= max(x3,x4) and min(y3,y4) <= y0 <= max(y3,y4):
            xy = Point(x0, y0)
        else:
            xy = None
        return xy

    def reduce_point(
        self,
        xcollapse = max, # function to collapse x values when line is horizontal
        ycollapse = lambda x:sum(x)/len(x), # function to collapse y values when line is vertical
        xycollapse = lambda x:sum(x)/len(x), # function to collapse x or y values when lines are sloped
        ):
        """
        Find a representative point on the line segment
        based on xcollapse, ycollapse, xycollapse functions

        Returns:
        - Point: representative point on the line segment
        """
        if self.A.x == self.B.x:
            return Point(self.A.x,ycollapse([self.A.y,self.B.y]))
        if self.A.y == self.B.y:
            return Point(xcollapse([self.A.x,self.B.x]),self.A.y)        
        return Point(xycollapse([self.A.x,self.B.x]),xycollapse([self.A.y,self.B.y]))

class Curve:
    """
    The Curve class implements a 2D curve

    Attributes:
        lines (list of Line) - the line segments
    Methods:
        __init__ - constructor
        __repr__ - string representation
        _intersect_lines - find intersections between two curves
        intersect - find intersections with another curve
        plot - plot the curve
    """

    def __init__(self,points=None,lines=None):
        """Create a curve

        Arguments:
        - points (list of Point): vertices of the curve
        - lines (list of Lines): line segments of the curve
        """
        points = [] if points is None else points
        self.lines = [] if lines is None else lines
        last = points[0] if len(points) > 0 else None
        for point in points:
            if point is not None and point != last:
                self.lines.append(Line(last,point))
            last = point
    
    def __repr__(self):
        return "Curve(" + ",".join([str(x) for x in self.lines]) + ")"

    def _intersect_lines(self, L1, n, other):
        result = []
        for m,L2 in enumerate(other.lines):
            P = L1.intersect(L2)
            if P:
                if isinstance(P, Line):
                    del_idx = []
                    for p,item in enumerate(result):
                        if isinstance(item[0], Point) and P.isin(item[0]):
                            del_idx.append(p)
                    for p in sorted(del_idx,reverse=True):
                        del result[p]
                    result.append((P,n,m)) # P is a line, n is the index of the line in self, m is the index of the line in other
                elif isinstance(P, Point):
                    found = False
                    for item in result:
                        if isinstance(item[0], Line) and item[0].isin(P):
                            found = True
                    if not found:
                        result.append((P,n,m))
                else:
                    raise ValueError(f"intersection {P} is not valid")
        return result
    
    def intersect(self,other,locate=False):
        """Find intersections with a curve

        Arguments:
        - other (Curve): other curve
        - locate (bool): flag to include location in return value

        Returns:
        - list of Point or Line: intersections
        - int: index to line on first curve
        - int: index to line on second curve
        """
        result = []
        for n,L1 in enumerate(self.lines):
            result_intersections = self._intersect_lines(L1, n, other)
            for intersection in result_intersections:
                if intersection[0] is not None:
                    if isinstance(intersection[0], Point):
                        found = False
                        for item in result:
                            if isinstance(item[0], Line) and item[0].isin(intersection[0]):
                                found = True
                        if not found:
                            result.append(intersection)
                    elif isinstance(intersection[0], Line):
                        del_idx = []
                        for item in result:
                            if isinstance(item[0], Point) and intersection[0].isin(item[0]):
                                del_idx.append(item)
                        for item in sorted(del_idx, reverse=True):
                            result.remove(item)
                        result.append(intersection)
        if not result:
            return None
        return result if locate else [x[0] for x in result]

    def plot(self,plt=plt,**kwargs):
        """Plot a curve

        Arguments:
        - plt (matplotlib.pyplot or figure): plotting context
        """
        X = []
        Y = []
        for L in self.lines:
            if len(X) == 0 or L.A.x != X[-1] or len(Y) == 0 or L.A.y != Y[-1]:
                X.append(L.A.x)
                Y.append(L.A.y)
            X.append(L.B.x)
            Y.append(L.B.y)
        plt.plot(X,Y,**kwargs)
        return plt

if __name__ == "__main__":

    import unittest
    import inspect
    import os
    def fname(fmt=None):
        if fmt:
            return fmt.format(fname=inspect.stack()[1][3])
        return inspect.stack()[1][3]

    os.makedirs("test", exist_ok=True)

    class TestGeometry(unittest.TestCase):

        def test_line_intersections(self):

            showall = False # shows all plots, not just flagged ones            
            def _test(L1,L2,P,plot):
                Q = L1.intersect(L2)
                if plot and Q != P:
                    plt.plot([L1.A.x,L1.B.x],[L1.A.y,L1.B.y],'-o',label='L1')
                    plt.plot([L2.A.x,L2.B.x],[L2.A.y,L2.B.y],'-o',label='L2')
                    if Q:
                        if isinstance(Q, list):
                            plt.plot([z.x for z in Q],[z.y for z in Q],'k--o')
                        else:
                            plt.plot(Q.x,Q.y,'ko')
                    if P:
                        if isinstance(P, list):
                            plt.plot([z.x for z in P],[z.y for z in P],'y--*')
                        else:
                            plt.plot(P.x,P.y,'y*')
                    if P != Q:
                        plt.title('No intersection')
                    plt.grid()
                    plt.legend()
                    plt.show()
                else:
                    self.assertEqual(Q,P)

            def test(L1, L2, P, plot=showall):                
                _test(L1,L2,P,plot)
                _test(L2,L1,P,plot)

            test(Line((1,1),(3,3)),Line((3,1),(1,3)),Point(2,2))
            test(Line((1,1),(3,3)),Line((3,4),(6,3)),None)

            test(Line((1,1),(3,3)),Line((2,2),(4,4)),Line((3.0,3.0),(2.0,2.0)))
            test(Line((2,2),(4,4)),Line((1,1),(3,3)),Line((3.0,3.0),(2.0,2.0)))
            test(Line((1,1),(3,3)),Line((4,4),(5,5)),None)
            test(Line((1,1),(3,3)),Line((4,4),(5,5)),None)

            test(Line((1,1),(1,3)),Line((1,2),(1,4)),Line((1.0,3.0), (1.0,2.0)))
            test(Line((1,1),(1,2)),Line((1,3),(1,4)),None)
            test(Line((1,1),(1,3)),Line((2,1),(2,3)),None)

            test(Line((1,1),(3,1)),Line((2,1),(4,1)),Line((3.0,1.0), (2.0,1.0)))
            test(Line((1,1),(2,1)),Line((3,1),(4,1)),None)
            test(Line((1,1),(3,1)),Line((1,2),(3,2)),None)

            test(Line((1,1),(2,1)),Line((0,0),(0,2)),None)
            test(Line((1,1),(2,1)),Line((1,0),(1,2)),Point(1,1))
            test(Line((1,1),(2,1)),Line((1.5,0),(1.5,2)),Point(1.5,1))
            test(Line((1,1),(2,1)),Line((2,0),(2,2)),Point(2,1))
            test(Line((1,1),(2,1)),Line((3,0),(3,2)),None)

            test(Line((1,1),(2,1)),Line((0,0),(0.5,2)),None)
            test(Line((1,1),(2,1)),Line((1,0),(1.5,2)),Point(1.25,1))
            test(Line((1,1),(2,1)),Line((1.5,0),(2,2)),Point(1.75,1))
            test(Line((1,1),(2,1)),Line((2,0),(2.5,2)),None)
            test(Line((1,1),(2,1)),Line((3,0),(3.5,2)),None)

            test(Line((1,1),(1,2)),Line((0,0),(2,0.5)),None)
            test(Line((1,1),(1,2)),Line((0,1),(2,1.5)),Point(1,1.25))
            test(Line((1,1),(1,2)),Line((0,1.5),(2,2)),Point(1,1.75))
            test(Line((1,1),(1,2)),Line((0,2),(2,2.5)),None)
            test(Line((1,1),(1,2)),Line((0,3),(2,3.5)),None)

        def test_curve_intersections(self):
            showall = False
            def _test(P1,P2,P,plot):
                C1,C2 = Curve(P1), Curve(P2)
                Q = C1.intersect(C2)
                Q = None if Q is None else Q[0]
                if plot or P != Q:
                    C1.plot(label='C1')
                    C2.plot(label='C2')
                    if Q:
                        plt.plot(Q.x,Q.y,'ko')
                    if P:
                        plt.plot(P.x,P.y,'w*')
                    if P != Q:
                        plt.title('No intersection')
                    plt.grid()
                    plt.legend()
                    plt.show()
                self.assertEqual(Q,P)
     
            def test(P1, P2, P,plot=showall):
                _test(P1, P2, P, plot)
                _test(P2, P1, P, plot)

            test([(0,0),(0,1),(1,1),(1,2)],[(0,0),(0,2)],Line((0,0),(0,1)))
            test([(0,0),(0,1),(1,1),(1,2)],[(0.5,0),(0.5,2)],Point(0.5,1.0))
            test([(0,0),(0,1),(1,1),(1,2)],[(1,0),(1,2)],Line((1,1),(1,2)))
            test([(0,0),(0,1),(1,1),(1,2)],[(1.5,0),(1.5,2)],None)

            test([(0,0),(0,2)],[(0,0),(0,1),(1,1),(1,2)],Line((0,0),(0,1)))
            test([(0.5,0),(0.5,2)],[(0,0),(0,1),(1,1),(1,2)],Point(0.5,1))
            test([(1,0),(1,2)],[(0,0),(0,1),(1,1),(1,2)],Line((1,1),(1,2)))
            test([(1.5,0),(1.5,2)],[(0,0),(0,1),(1,1),(1,2)],None)

    unittest.main()
