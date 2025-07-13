import pygame
import random
import math
import sys
from typing import List, Tuple, Set

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
CITY_RADIUS = 2
NUM_CITIES = 75
PANEL_WIDTH = SCREEN_WIDTH // 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

class City:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
    
    def distance_to(self, other: 'City') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def copy_with_offset(self, offset_x: float):
        return City(self.x + offset_x, self.y, self.id)

def generate_cities(num_cities: int) -> List[City]:
    """Generate random cities within the panel bounds"""
    cities = []
    margin = 50
    for i in range(num_cities):
        x = random.randint(margin, PANEL_WIDTH - margin)
        y = random.randint(margin, SCREEN_HEIGHT - margin - 100)
        cities.append(City(x, y, i))
    return cities

def cross_product(o: City, a: City, b: City) -> float:
    """Calculate cross product for convex hull"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def convex_hull(cities: List[City]) -> List[City]:
    """Graham scan algorithm to find convex hull"""
    if len(cities) < 3:
        return cities
    
    start = min(cities, key=lambda p: (p.y, p.x))
    
    def polar_angle(city):
        dx = city.x - start.x
        dy = city.y - start.y
        return math.atan2(dy, dx)
    
    sorted_cities = sorted([city for city in cities if city != start], key=polar_angle)
    
    hull = [start]
    for city in sorted_cities:
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], city) <= 0:
            hull.pop()
        hull.append(city)
    
    return hull

def point_to_line_distance(point: City, line_start: City, line_end: City) -> float:
    """Calculate distance from point to line segment"""
    A = point.x - line_start.x
    B = point.y - line_start.y
    C = line_end.x - line_start.x
    D = line_end.y - line_start.y
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0: return point.distance_to(line_start)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = line_start.x, line_start.y
    elif param > 1:
        xx, yy = line_end.x, line_end.y
    else:
        xx = line_start.x + param * C
        yy = line_start.y + param * D
    
    dx = point.x - xx
    dy = point.y - yy
    return math.sqrt(dx * dx + dy * dy)

def calculate_total_distance(path: List[City]) -> float:
    """Calculate total distance of a path"""
    if len(path) < 2: return 0
    total = 0
    for i in range(len(path)):
        total += path[i].distance_to(path[(i + 1) % len(path)])
    return total

def two_opt_improve(path: List[City]) -> List[City]:
    """Apply 2-opt improvement to eliminate crossing edges"""
    if len(path) < 4: return path
    
    best_path = path[:]
    improved = True
    while improved:
        improved = False
        best_distance = calculate_total_distance(best_path)
        for i in range(len(best_path) - 1):
            for j in range(i + 2, len(best_path)):
                if i == 0 and j == len(best_path) - 1: continue
                
                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                new_distance = calculate_total_distance(new_path)
                
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return best_path

class TSPSolver:
    def __init__(self, cities: List[City]):
        self.original_cities = cities
        self.reset()
    
    def reset(self):
        self.cities = self.original_cities[:]
        self.path = []
        self.visited = set()
        self.current_city = None
        self.finished = False
        self.optimized = False
        self.original_path = None
        self.step = 0

class NearestNeighborSolver(TSPSolver):
    def __init__(self, cities: List[City]):
        super().__init__(cities)
        self.unvisited = set(cities)
        self.current_city = cities[0]
        self.path = [self.current_city]
        self.visited = {self.current_city}
        self.unvisited.remove(self.current_city)
        self.candidate_city = None
    
    def step_forward(self):
        if self.finished: return
        
        if not self.unvisited:
            self.finished = True
            if not self.optimized:
                self.original_path = self.path[:]
                self.path = two_opt_improve(self.path)
                self.optimized = True
            return
        
        self.candidate_city = min(self.unvisited, key=lambda city: self.current_city.distance_to(city))
        self.path.append(self.candidate_city)
        self.visited.add(self.candidate_city)
        self.unvisited.remove(self.candidate_city)
        self.current_city = self.candidate_city
        self.step += 1

class ConvexHullSolver(TSPSolver):
    def __init__(self, cities: List[City]):
        super().__init__(cities)
        self.hull = convex_hull(cities)
        self.path = self.hull[:]
        self.visited = set(self.hull)
        self.remaining = [city for city in cities if city not in self.hull]
        self.candidate_city = None
        self.best_edge = None
    
    def step_forward(self):
        if self.finished: return

        if not self.remaining:
            self.finished = True
            if not self.optimized:
                self.original_path = self.path[:]
                self.path = two_opt_improve(self.path)
                self.optimized = True
            return
        
        best_city = None
        best_distance = float('inf')
        best_insert_pos = 0
        best_line_start, best_line_end = None, None
        
        for city in self.remaining:
            for i in range(len(self.path)):
                line_start = self.path[i]
                line_end = self.path[(i + 1) % len(self.path)]
                distance = point_to_line_distance(city, line_start, line_end)
                if distance < best_distance:
                    best_distance = distance
                    best_city = city
                    best_insert_pos = i + 1
                    best_line_start, best_line_end = line_start, line_end
        
        self.candidate_city = best_city
        self.best_edge = (best_line_start, best_line_end)
        
        if best_city:
            self.path.insert(best_insert_pos, best_city)
            self.visited.add(best_city)
            self.remaining.remove(best_city)
        
        self.step += 1

class HybridSolver(TSPSolver):
    def __init__(self, cities: List[City]):
        super().__init__(cities)
        self.switch_point = len(cities) // 1.6
        self.unvisited = set(cities)
        self.current_city = cities[0]
        self.path = [self.current_city]
        self.visited = {self.current_city}
        self.unvisited.remove(self.current_city)
        self.candidate_city = None
        self.best_edge = None

    def step_forward(self):
        if self.finished: return
        
        if not self.unvisited:
            self.finished = True
            if not self.optimized:
                self.original_path = self.path[:]
                self.path = two_opt_improve(self.path)
                self.optimized = True
            return

        # Phase 1: Nearest Neighbor
        if len(self.path) < self.switch_point:
            self.best_edge = None # Not used in this phase
            self.candidate_city = min(self.unvisited, key=lambda city: self.current_city.distance_to(city))
            self.path.append(self.candidate_city)
            self.visited.add(self.candidate_city)
            self.unvisited.remove(self.candidate_city)
            self.current_city = self.candidate_city
        
        # Phase 2: Convex Hull-style Insertion
        else:
            self.current_city = None # Not used in this phase
            best_city_to_insert = None
            min_insertion_cost = float('inf')
            best_insert_pos = 0
            
            for city in self.unvisited:
                for i in range(len(self.path)):
                    prev_city = self.path[i]
                    next_city = self.path[(i + 1) % len(self.path)]
                    
                    # Cost: dist(prev, city) + dist(city, next) - dist(prev, next)
                    cost = prev_city.distance_to(city) + city.distance_to(next_city) - prev_city.distance_to(next_city)
                    
                    if cost < min_insertion_cost:
                        min_insertion_cost = cost
                        best_city_to_insert = city
                        best_insert_pos = i + 1
            
            self.candidate_city = best_city_to_insert
            if best_city_to_insert:
                self.path.insert(best_insert_pos, best_city_to_insert)
                self.visited.add(best_city_to_insert)
                self.unvisited.remove(best_city_to_insert)
        
        self.step += 1

def draw_cities(screen, cities: List[City], visited: Set[City], current: City = None, candidate: City = None):
    for city in cities:
        color = WHITE
        if city == current: color = YELLOW
        elif city == candidate: color = ORANGE
        elif city in visited: color = GREEN
        
        pygame.draw.circle(screen, color, (int(city.x), int(city.y)), CITY_RADIUS)

def draw_path(screen, path: List[City], color: Tuple[int, int, int]):
    if len(path) > 1:
        pygame.draw.lines(screen, color, True, [(c.x, c.y) for c in path], 2)

def draw_panel(screen, solver, title, color, x_offset):
    # Draw title
    title_font = pygame.font.Font(None, 32)
    title_text = title_font.render(title, True, color)
    screen.blit(title_text, (x_offset + 10, 10))

    # Draw path and cities
    draw_path(screen, solver.path, color)
    draw_cities(screen, solver.cities, solver.visited, solver.current_city, solver.candidate_city)

    # Draw special indicators for insertion algorithms
    if isinstance(solver, ConvexHullSolver) and solver.candidate_city and solver.best_edge:
        start, end = solver.best_edge
        pygame.draw.line(screen, ORANGE, (start.x, start.y), (end.x, end.y), 3)
    
    # Draw stats
    font = pygame.font.Font(None, 24)
    remaining_cities = len(solver.unvisited) if hasattr(solver, 'unvisited') else len(solver.remaining) if hasattr(solver, 'remaining') else 0
    
    stats = [
        f"Step: {solver.step}",
        f"Remaining: {remaining_cities}",
    ]
    if solver.finished and solver.optimized:
        original_dist = calculate_total_distance(solver.original_path) if solver.original_path else 0
        final_dist = calculate_total_distance(solver.path)
        stats.append(f"Original: {original_dist:.1f}")
        stats.append(f"2-opt: {final_dist:.1f}")

    for i, stat in enumerate(stats):
        text = font.render(stat, True, WHITE)
        screen.blit(text, (x_offset + 10, 40 + i * 20))


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("TSP Algorithm Comparison")
    clock = pygame.time.Clock()
    
    # Generate one set of cities
    base_cities = generate_cities(NUM_CITIES)
    
    # Create solvers for each panel
    nn_solver = NearestNeighborSolver([c.copy_with_offset(0) for c in base_cities])
    ch_solver = ConvexHullSolver([c.copy_with_offset(PANEL_WIDTH) for c in base_cities])
    hybrid_solver = HybridSolver([c.copy_with_offset(PANEL_WIDTH * 2) for c in base_cities])
    
    solvers = [nn_solver, ch_solver, hybrid_solver]
    titles = ["Nearest Neighbor", "Convex Hull", "Hybrid (NN -> Insert)"]
    colors = [RED, BLUE, PURPLE]
    
    running = True
    auto_step = True
    step_delay = 50
    last_step_time = 0
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_step = not auto_step
                elif event.key == pygame.K_r:
                    base_cities = generate_cities(NUM_CITIES)
                    nn_solver = NearestNeighborSolver([c.copy_with_offset(0) for c in base_cities])
                    ch_solver = ConvexHullSolver([c.copy_with_offset(PANEL_WIDTH) for c in base_cities])
                    hybrid_solver = HybridSolver([c.copy_with_offset(PANEL_WIDTH * 2) for c in base_cities])
                    solvers = [nn_solver, ch_solver, hybrid_solver]
                elif event.key == pygame.K_RIGHT:
                    for solver in solvers:
                        if not solver.finished:
                            solver.step_forward()

        # CORRECTED AUTO-STEP LOGIC
        if auto_step and (current_time - last_step_time > step_delay):
            stepped_this_frame = False
            for solver in solvers:
                if not solver.finished:
                    solver.step_forward()
                    stepped_this_frame = True
            if stepped_this_frame:
                last_step_time = current_time

        screen.fill(BLACK)
        
        # Draw panels and dividing lines
        for i, solver in enumerate(solvers):
            x_offset = i * PANEL_WIDTH
            draw_panel(screen, solver, titles[i], colors[i], x_offset)
            if i > 0:
                pygame.draw.line(screen, WHITE, (x_offset, 0), (x_offset, SCREEN_HEIGHT), 2)
        
        # Draw instructions at the bottom
        font = pygame.font.Font(None, 24)
        instructions = [
            f"Auto-step: {'ON' if auto_step else 'OFF'} [SPACE to toggle]",
            "[R] to Reset with New Cities",
            "[RIGHT ARROW] for Manual Step"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, GRAY)
            screen.blit(text, (10, SCREEN_HEIGHT - 80 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
