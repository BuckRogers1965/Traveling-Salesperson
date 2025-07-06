import pygame
import random
import math
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class City:
    x: float
    y: float
    name: str

class TSPVisualizer:
    def __init__(self, width=1400, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TSP Algorithm Comparison with Path Crossing Fixes")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Algorithm states
        self.cities = []
        self.distance_matrix = []
        self.greedy_paths = []
        self.nearest_neighbor_path = []
        self.greedy_tour = []
        self.nn_tour = []
        
        # Completion states
        self.greedy_complete = False
        self.nn_complete = False
        self.greedy_closed = False
        self.nn_closed = False
        self.greedy_optimized = False
        self.nn_optimized = False
        
        # Timing and metrics
        self.greedy_time = 0
        self.nn_time = 0
        self.greedy_distance = 0
        self.nn_distance = 0
        self.greedy_optimized_distance = 0
        self.nn_optimized_distance = 0
        self.greedy_improvements = 0
        self.nn_improvements = 0
        
        # Animation control
        self.animation_speed = 50  # milliseconds
        self.running = False
        self.optimization_step = 0
        
    def generate_cities(self, num_cities=500):
        """Generate random cities"""
        self.cities = []
        margin = 50
        
        for i in range(num_cities):
            x = random.randint(margin, self.width//2 - margin)
            y = random.randint(margin + 120, self.height - margin)
            self.cities.append(City(x, y, f"C{i}"))
        
        # Build distance matrix
        self.build_distance_matrix()
        
    def build_distance_matrix(self):
        """Build complete distance matrix"""
        n = len(self.cities)
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.cities[i].x - self.cities[j].x
                    dy = self.cities[i].y - self.cities[j].y
                    self.distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    
    def get_path_distance(self, path: List[int]) -> float:
        """Calculate total distance of a path"""
        total = 0
        for i in range(len(path) - 1):
            total += self.distance_matrix[path[i]][path[i + 1]]
        return total
    
    def get_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour (including return to start)"""
        total = 0
        for i in range(len(tour)):
            next_city = (i + 1) % len(tour)
            total += self.distance_matrix[tour[i]][tour[next_city]]
        return total
    
    def greedy_path_merger_step(self):
        """
        One step of the greedy path merger algorithm.
        Modified to strictly prioritize connecting to individual 'free' cities (paths of length 1)
        before merging longer existing paths.
        """
        n = len(self.cities)

        # Phase 0: Initialization - Each city is initially a path of length 1
        if not self.greedy_paths:
            self.greedy_paths = [[i] for i in range(n)]
            return True

        # Check for completion: only one path left
        if len(self.greedy_paths) <= 1:
            if not self.greedy_complete:
                self.greedy_complete = True
                if self.greedy_paths: # Ensure there's a path if n > 0
                    self.greedy_distance = self.get_path_distance(self.greedy_paths[0])
                else: # Handle case of 0 cities
                    self.greedy_distance = 0
            return False

        min_dist = float('inf')
        best_action = None # Will store details for either 'extend' or 'merge'

        # --- Strategy 1: Prioritize connecting a single-city path (a "free point") to an existing path's endpoint ---
        # Find the shortest connection from an existing path's endpoint to a free single city (path of length 1).
        
        # Iterate through all current paths (these are the paths we might extend)
        for p1_idx, path1 in enumerate(self.greedy_paths):
            # Consider both ends of path1
            path1_endpoints = [path1[0]]
            if len(path1) > 1: # If path has more than one city, it has two distinct ends
                path1_endpoints.append(path1[-1])

            # Iterate through all other paths (path2) to find single-city paths to absorb
            for p2_idx, path2 in enumerate(self.greedy_paths):
                if p1_idx == p2_idx: continue # Don't connect a path to itself

                # We are looking for 'free points', which are paths of length 1
                if len(path2) == 1:
                    target_single_city_node = path2[0] # The city index of the free point

                    for end_node_from_path1 in path1_endpoints:
                        dist = self.distance_matrix[end_node_from_path1][target_single_city_node]
                        if dist < min_dist:
                            min_dist = dist
                            best_action = {
                                'type': 'extend',
                                'path_idx_to_extend': p1_idx, # The path that will grow
                                'endpoint_node': end_node_from_path1, # The end of path_to_extend
                                'single_city_path_idx_to_consume': p2_idx, # The index of the path [target_single_city_node]
                                'target_city': target_single_city_node # The city being absorbed
                            }
        
        # --- Strategy 2: If no single-city path can be connected (or if it's cheaper to merge already-formed paths) ---
        # Only consider merging two existing paths if no 'extend' action was found, or if a merge is globally shorter.
        # The user's request implies *strict* prioritization of 'extend'. So, only if best_action is None.
        
        if best_action is None: # No single-city path was found to connect to
            # This means all cities are already part of multi-city paths, or there's only one path remaining.
            # So, now we must merge existing paths.
            for i in range(len(self.greedy_paths)):
                for j in range(i + 1, len(self.greedy_paths)):
                    path_a = self.greedy_paths[i]
                    path_b = self.greedy_paths[j]
                    
                    # Get endpoints for both paths
                    endpoints_a = [path_a[0]]
                    if len(path_a) > 1: endpoints_a.append(path_a[-1])
                    
                    endpoints_b = [path_b[0]]
                    if len(path_b) > 1: endpoints_b.append(path_b[-1])
                    
                    for end_a in endpoints_a:
                        for end_b in endpoints_b:
                            dist = self.distance_matrix[end_a][end_b]
                            if dist < min_dist: # Find the shortest merge
                                min_dist = dist
                                best_action = {
                                    'type': 'merge',
                                    'path_a_idx': i,
                                    'path_b_idx': j,
                                    'end_a_node': end_a,
                                    'end_b_node': end_b
                                }
        
        # --- Execute the best action found ---
        if best_action is None:
            # This should only be reached if len(greedy_paths) <= 1, which is handled at the start.
            # It also acts as a safeguard if somehow no connection could be found (e.g., 0 cities).
            return False 
        
        if best_action['type'] == 'extend':
            p1_idx = best_action['path_idx_to_extend']
            endpoint_node = best_action['endpoint_node']
            p2_idx_to_consume = best_action['single_city_path_idx_to_consume']
            target_city = best_action['target_city']
            
            path_to_extend = self.greedy_paths[p1_idx]
            
            # Add target_city to the correct end of the path_to_extend
            if path_to_extend[0] == endpoint_node: # Connect to the start
                path_to_extend.insert(0, target_city)
            else: # Connect to the end
                path_to_extend.append(target_city)
            
            # Remove the consumed single-city path (path2) from the list
            # Rebuilding the list is safer than `pop` or `del` with changing indices.
            new_paths = []
            for idx, path in enumerate(self.greedy_paths):
                if idx != p2_idx_to_consume: # Keep all paths except the one consumed
                    new_paths.append(path)
            self.greedy_paths = new_paths
                
        elif best_action['type'] == 'merge':
            path_a_idx = best_action['path_a_idx']
            path_b_idx = best_action['path_b_idx']
            end_a = best_action['end_a_node']
            end_b = best_action['end_b_node']
            
            # Retrieve paths by index. Make a copy to avoid modifying references during iteration.
            path_a = list(self.greedy_paths[path_a_idx])
            path_b = list(self.greedy_paths[path_b_idx])
            
            # Orient paths correctly for concatenation based on which ends are connecting
            if path_a[-1] != end_a: path_a.reverse()
            if path_b[0] != end_b: path_b.reverse()
            
            merged_path = path_a + path_b
            
            # Create a new list of paths, excluding the two merged ones and adding the new one
            new_paths = []
            # Use a set for efficient lookup of indices to exclude
            indices_to_exclude = {path_a_idx, path_b_idx} 
            for idx, path in enumerate(self.greedy_paths):
                if idx not in indices_to_exclude:
                    new_paths.append(path)
            new_paths.append(merged_path)
            self.greedy_paths = new_paths
        
        return True # An operation was successfully performed

    def nearest_neighbor_step(self):
        """One step of nearest neighbor algorithm"""
        if not self.nearest_neighbor_path:
            # Start with random city
            self.nearest_neighbor_path = [0]
            return True
        
        if len(self.nearest_neighbor_path) >= len(self.cities):
            if not self.nn_complete:
                self.nn_complete = True
                self.nn_distance = self.get_path_distance(self.nearest_neighbor_path)
            return False
        
        # Find nearest unvisited city
        current_city = self.nearest_neighbor_path[-1]
        visited = set(self.nearest_neighbor_path)
        
        min_dist = float('inf')
        nearest_city = None
        
        for i in range(len(self.cities)):
            if i not in visited:
                dist = self.distance_matrix[current_city][i]
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = i
        
        if nearest_city is not None:
            self.nearest_neighbor_path.append(nearest_city)
        
        return True
    
    def close_tours(self):
        """Close both tours by connecting endpoints"""
        if self.greedy_complete and not self.greedy_closed:
            if self.greedy_paths:
                self.greedy_tour = self.greedy_paths[0][:]
                self.greedy_closed = True
        
        if self.nn_complete and not self.nn_closed:
            self.nn_tour = self.nearest_neighbor_path[:]
            self.nn_closed = True
    
    def lines_intersect(self, p1, p2, p3, p4):
        """Check if line segments p1-p2 and p3-p4 intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def get_city_pos(self, city_idx):
        """Get city position as tuple"""
        return (self.cities[city_idx].x, self.cities[city_idx].y)
    
    def two_opt_step(self, tour, improvements_count):
        """Perform one step of 2-opt optimization"""
        n = len(tour)
        
        # Check a small batch of edge pairs per step for real-time visualization
        batch_size = 10
        start_i = (self.optimization_step * batch_size) % (n - 1)
        
        for i in range(start_i, min(start_i + batch_size, n - 1)):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:  # Don't check the same edge (connecting last to first)
                    continue
                
                # Get the four cities involved
                city_a = tour[i]
                city_b = tour[i + 1]
                city_c = tour[j]
                city_d = tour[(j + 1) % n]
                
                # Check if edges cross (geometric check)
                pos_a = self.get_city_pos(city_a)
                pos_b = self.get_city_pos(city_b)
                pos_c = self.get_city_pos(city_c)
                pos_d = self.get_city_pos(city_d)
                
                if self.lines_intersect(pos_a, pos_b, pos_c, pos_d):
                    # Calculate current distance (sum of the two crossing edges)
                    current_dist = (self.distance_matrix[city_a][city_b] + 
                                  self.distance_matrix[city_c][city_d])
                    
                    # Calculate new distance after swap (sum of the two new edges)
                    new_dist = (self.distance_matrix[city_a][city_c] + 
                              self.distance_matrix[city_b][city_d])
                    
                    if new_dist < current_dist:
                        # Perform 2-opt swap: reverse the segment between i+1 and j
                        new_tour = tour[:]
                        new_tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        
                        improvements_count += 1
                        self.optimization_step += 1
                        return new_tour, improvements_count, True # Improvement made
        
        self.optimization_step += 1
        return tour, improvements_count, False # No improvement in this batch
    
    def optimize_tours(self):
        """Perform 2-opt optimization on both tours"""
        # Reset optimization_step for each tour's optimization state
        # The current implementation uses one shared optimization_step, which might
        # cause slow completion detection if one finishes much earlier.
        # For simple animation, it's fine.
        
        if self.greedy_closed and not self.greedy_optimized:
            new_tour, self.greedy_improvements, improved = self.two_opt_step(
                self.greedy_tour, self.greedy_improvements)
            
            if improved:
                self.greedy_tour = new_tour
                # >>> START OF THE ONLY CHANGE FOR GREEDY ALGORITHM <<<
                self.greedy_distance = self.get_tour_distance(self.greedy_tour) 
                # >>> END OF THE ONLY CHANGE FOR GREEDY ALGORITHM <<<
            else:
                # This completion check is heuristic; a full pass requires optimization_step to reach n*n-like values.
                # For animation, we can just say it's "done" if no improvements for a while
                # or after a certain number of iterations relative to n^2.
                # For now, if no improvement in this step, assume it's stable for current iteration batch.
                # A more robust check would involve iterating until no improvement in a full pass.
                if self.optimization_step % (len(self.greedy_tour) * (len(self.greedy_tour) - 1)) == 0 and not improved:
                    self.greedy_optimized = True
                    self.greedy_optimized_distance = self.get_tour_distance(self.greedy_tour)
                
        if self.nn_closed and not self.nn_optimized:
            new_tour, self.nn_improvements, improved = self.two_opt_step(
                self.nn_tour, self.nn_improvements)
            
            if improved:
                self.nn_tour = new_tour
                # >>> START OF THE ONLY CHANGE FOR NEAREST NEIGHBOR ALGORITHM <<<
                self.nn_distance = self.get_tour_distance(self.nn_tour)
                # >>> END OF THE ONLY CHANGE FOR NEAREST NEIGHBOR ALGORITHM <<<
            else:
                if self.optimization_step % (len(self.nn_tour) * (len(self.nn_tour) - 1)) == 0 and not improved:
                    self.nn_optimized = True
                    self.nn_optimized_distance = self.get_tour_distance(self.nn_tour)
    
    def draw_cities(self):
        """Draw all cities"""
        for i, city in enumerate(self.cities):
            # Draw city as small circle
            pygame.draw.circle(self.screen, self.BLACK, (int(city.x), int(city.y)), 3)
    
    def draw_greedy_paths(self):
        """Draw current greedy paths or tour"""
        if self.greedy_tour:
            # Draw complete tour
            color = self.GREEN if self.greedy_optimized else self.RED
            for i in range(len(self.greedy_tour)):
                next_i = (i + 1) % len(self.greedy_tour)
                city_a = self.cities[self.greedy_tour[i]]
                city_b = self.cities[self.greedy_tour[next_i]]
                pygame.draw.line(self.screen, color, 
                               (int(city_a.x), int(city_a.y)), 
                               (int(city_b.x), int(city_b.y)), 2)
        else:
            # Draw current paths
            colors = [self.RED, self.BLUE, self.GREEN, self.PURPLE, self.YELLOW, self.ORANGE, self.GRAY]
            
            for i, path in enumerate(self.greedy_paths):
                color = colors[i % len(colors)]
                if len(path) > 1: # Draw connections within multi-city paths
                    for j in range(len(path) - 1):
                        city_a = self.cities[path[j]]
                        city_b = self.cities[path[j + 1]]
                        pygame.draw.line(self.screen, color, 
                                       (int(city_a.x), int(city_a.y)), 
                                       (int(city_b.x), int(city_b.y)), 2)
                # Mark single cities differently or just let draw_cities handle it
                # For single cities, path is [city_idx], which is not drawn by loop above.
                # They are effectively drawn by draw_cities()
                
    def draw_nearest_neighbor_path(self):
        """Draw current nearest neighbor path or tour"""
        x_offset = self.width // 2
        
        if self.nn_tour:
            # Draw complete tour
            color = self.GREEN if self.nn_optimized else self.BLUE
            for i in range(len(self.nn_tour)):
                next_i = (i + 1) % len(self.nn_tour)
                city_a = self.cities[self.nn_tour[i]]
                city_b = self.cities[self.nn_tour[next_i]]
                pygame.draw.line(self.screen, color, 
                               (int(city_a.x + x_offset), int(city_a.y)), 
                               (int(city_b.x + x_offset), int(city_b.y)), 2)
        else:
            # Draw current path
            if len(self.nearest_neighbor_path) > 1:
                for i in range(len(self.nearest_neighbor_path) - 1):
                    city_a = self.cities[self.nearest_neighbor_path[i]]
                    city_b = self.cities[self.nearest_neighbor_path[i + 1]]
                    pygame.draw.line(self.screen, self.BLUE, 
                                   (int(city_a.x + x_offset), int(city_a.y)), 
                                   (int(city_b.x + x_offset), int(city_b.y)), 2)
    
    def draw_cities_right(self):
        """Draw cities on right side for nearest neighbor"""
        x_offset = self.width // 2
        for i, city in enumerate(self.cities):
            # For NN, color visited cities differently if path is not complete
            color = self.BLACK
            if not self.nn_tour:
                color = self.ORANGE if i in self.nearest_neighbor_path else self.BLACK
            pygame.draw.circle(self.screen, color, 
                             (int(city.x + x_offset), int(city.y)), 3)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Title
        title = self.font.render("TSP Comparison with 2-Opt Optimization", True, self.BLACK)
        self.screen.blit(title, (10, 10))
        
        # Left side - Greedy Path Merger
        left_title = self.font.render("Greedy Path Grower (Custom)", True, self.BLACK)
        self.screen.blit(left_title, (10, 40))
        
        y_pos = 65
        
        # Count of single (isolated) cities for the custom greedy
        single_paths_count = sum(1 for path in self.greedy_paths if len(path) == 1)
        if not self.greedy_complete:
            paths_text = f"Paths: {len(self.greedy_paths)} (Isolated: {single_paths_count})"
            paths_surface = self.small_font.render(paths_text, True, self.BLACK)
            self.screen.blit(paths_surface, (10, y_pos))
            y_pos += 20
        
        if self.greedy_complete:
            if not self.greedy_closed:
                status_text = "Closing tour..."
            elif not self.greedy_optimized:
                status_text = f"Optimizing... (fixes: {self.greedy_improvements})"
            else:
                status_text = "Complete!"
            
            status_surface = self.small_font.render(status_text, True, self.BLACK)
            self.screen.blit(status_surface, (10, y_pos))
            y_pos += 20
            
            # This line will now update live because self.greedy_distance is updated in optimize_tours
            initial_dist = f"Initial: {self.greedy_distance:.1f}" 
            initial_surface = self.small_font.render(initial_dist, True, self.BLACK)
            self.screen.blit(initial_surface, (10, y_pos))
            y_pos += 20
            
            if self.greedy_optimized:
                final_dist = f"Optimized: {self.greedy_optimized_distance:.1f}"
                final_surface = self.small_font.render(final_dist, True, self.GREEN)
                self.screen.blit(final_surface, (10, y_pos))
                y_pos += 20
                
                improvement = ((self.greedy_distance - self.greedy_optimized_distance) / 
                             self.greedy_distance * 100)
                improv_text = f"Improved: {improvement:.1f}%"
                improv_surface = self.small_font.render(improv_text, True, self.GREEN)
                self.screen.blit(improv_surface, (10, y_pos))
        
        # Right side - Nearest Neighbor
        right_title = self.font.render("Nearest Neighbor", True, self.BLACK)
        self.screen.blit(right_title, (self.width // 2 + 10, 40))
        
        y_pos = 65
        
        if not self.nn_complete:
            visited_text = f"Visited: {len(self.nearest_neighbor_path)}/{len(self.cities)}"
            visited_surface = self.small_font.render(visited_text, True, self.BLACK)
            self.screen.blit(visited_surface, (self.width // 2 + 10, y_pos))
            y_pos += 20
        
        if self.nn_complete:
            if not self.nn_closed:
                status_text = "Closing tour..."
            elif not self.nn_optimized:
                status_text = f"Optimizing... (fixes: {self.nn_improvements})"
            else:
                status_text = "Complete!"
            
            status_surface = self.small_font.render(status_text, True, self.BLACK)
            self.screen.blit(status_surface, (self.width // 2 + 10, y_pos))
            y_pos += 20
            
            # This line will now update live because self.nn_distance is updated in optimize_tours
            initial_dist = f"Initial: {self.nn_distance:.1f}" 
            initial_surface = self.small_font.render(initial_dist, True, self.BLACK)
            self.screen.blit(initial_surface, (self.width // 2 + 10, y_pos))
            y_pos += 20
            
            if self.nn_optimized:
                final_dist = f"Optimized: {self.nn_optimized_distance:.1f}"
                final_surface = self.small_font.render(final_dist, True, self.GREEN)
                self.screen.blit(final_surface, (self.width // 2 + 10, y_pos))
                y_pos += 20
                
                improvement = ((self.nn_distance - self.nn_optimized_distance) / 
                             self.nn_distance * 100)
                improv_text = f"Improved: {improvement:.1f}%"
                improv_surface = self.small_font.render(improv_text, True, self.GREEN)
                self.screen.blit(improv_surface, (self.width // 2 + 10, y_pos))
        
        # Divider line
        pygame.draw.line(self.screen, self.GRAY, 
                        (self.width // 2, 0), (self.width // 2, self.height), 2)
        
        # Instructions
        if not self.running:
            inst_text = "Press SPACE to start/restart, Q to quit"
            inst_surface = self.small_font.render(inst_text, True, self.BLACK)
            self.screen.blit(inst_surface, (10, self.height - 25))
        
        # Legend
        legend_y = self.height - 80
        legend_items = [
            ("Building paths (Left)", self.RED),
            ("Building path (Right)", self.BLUE),
            ("Optimized tour", self.GREEN)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            pygame.draw.circle(self.screen, color, (20 + i * 150, legend_y), 5)
            text_surface = self.small_font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (30 + i * 150, legend_y - 8))
    
    def reset(self):
        """Reset algorithm states"""
        self.greedy_paths = []
        self.nearest_neighbor_path = []
        self.greedy_tour = []
        self.nn_tour = []
        self.greedy_complete = False
        self.nn_complete = False
        self.greedy_closed = False
        self.nn_closed = False
        self.greedy_optimized = False
        self.nn_optimized = False
        self.greedy_time = 0
        self.nn_time = 0
        self.greedy_distance = 0
        self.nn_distance = 0
        self.greedy_optimized_distance = 0
        self.nn_optimized_distance = 0
        self.greedy_improvements = 0
        self.nn_improvements = 0
        self.running = False
        self.optimization_step = 0 # Reset optimization step on reset
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        self.generate_cities(500) # Start with 500 cities
        
        last_update = time.time()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_SPACE:
                        self.reset()
                        self.generate_cities(500) # Re-generate cities on restart
                        self.running = True
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Update algorithms
            if self.running:
                current_time = time.time()
                
                if current_time - last_update >= self.animation_speed / 1000.0:
                    # Update greedy path merger
                    if not self.greedy_complete:
                        self.greedy_path_merger_step()
                    
                    # Update nearest neighbor
                    if not self.nn_complete:
                        self.nearest_neighbor_step()
                    
                    # Close tours when both algorithms complete their initial path building
                    if self.greedy_complete and self.nn_complete:
                        self.close_tours()
                    
                    # Run optimization
                    if self.greedy_closed or self.nn_closed:
                        self.optimize_tours()
                    
                    # Check if everything is complete
                    if self.greedy_optimized and self.nn_optimized:
                        self.running = False
                    
                    last_update = current_time
            
            # Draw everything
            self.draw_cities()
            self.draw_cities_right() # Draw cities on the right side for NN
            self.draw_greedy_paths()
            self.draw_nearest_neighbor_path()
            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(120) # Cap frame rate

        pygame.quit()

if __name__ == "__main__":
    visualizer = TSPVisualizer()
    visualizer.run()
