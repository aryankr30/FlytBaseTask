"""
UAV Strategic Deconfliction System
==================================

This module implements an enhanced 3D strategic deconfliction system for drone operations
in shared airspace. It checks for conflicts in space and time between a primary
drone mission and other drones' flight paths across their entire interpolated trajectories.

Author: Harshit Patel
Date: May 12, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
import time
from dataclasses import dataclass
from scipy.spatial import distance
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
from datetime import datetime, timedelta


@dataclass
class Waypoint:
    """Class representing a waypoint with spatial coordinates and optional altitude."""
    x: float
    y: float
    z: float = 0.0  # Default to 0 for 2D, include for 3D support


@dataclass
class TimeWindow:
    """Class representing a time window with start and end times."""
    start: float  # Time in seconds from reference point
    end: float    # Time in seconds from reference point


@dataclass
class DroneState:
    """Class representing the state of a drone at a specific time."""
    x: float
    y: float
    z: float
    time: float


@dataclass
class WaypointMission:
    """Class representing a waypoint mission for a drone."""
    waypoints: List[Waypoint]
    time_window: TimeWindow


@dataclass
class SimulatedFlight:
    """Class representing a simulated flight with waypoints and specific timings."""
    drone_id: str
    states: List[DroneState]  # Time-specific drone states


class DeconflictionSystem:
    """
    Strategic deconfliction system for verifying the safety of drone missions
    in shared airspace with multiple other drones.
    """
    
    def __init__(self, safety_buffer: float = 10.0, interpolation_resolution: int = 200):
        """
        Initialize the deconfliction system.
        
        Args:
            safety_buffer: Minimum safe distance between drones in meters
            interpolation_resolution: Number of points to use for trajectory interpolation
        """
        self.safety_buffer = safety_buffer
        self.interpolation_resolution = interpolation_resolution
        self.simulated_flights: List[SimulatedFlight] = []
        
    def register_simulated_flights(self, flights: List[SimulatedFlight]) -> None:
        """
        Register the flight schedules of other drones in the system.
        
        Args:
            flights: List of simulated flights to register
        """
        self.simulated_flights = flights
        
    def _interpolate_trajectory(self, waypoints: List[Waypoint], 
                                time_window: TimeWindow,
                                num_points: int = None) -> List[DroneState]:
        """
        Interpolate a trajectory between waypoints with straight line segments.
        
        Args:
            waypoints: List of waypoints
            time_window: Time window for the mission
            num_points: Number of interpolated points (defaults to self.interpolation_resolution)
            
        Returns:
            List of drone states representing the interpolated trajectory
        """
        if num_points is None:
            num_points = self.interpolation_resolution
            
        if len(waypoints) < 2:
            return [DroneState(waypoints[0].x, waypoints[0].y, waypoints[0].z, time_window.start)]
        
        # Generate interpolated trajectory with linear segments
        trajectory = []
        total_time = time_window.end - time_window.start
        
        # Calculate segment lengths and total distance
        distances = []
        total_distance = 0
        for i in range(1, len(waypoints)):
            dist = distance.euclidean(
                (waypoints[i-1].x, waypoints[i-1].y, waypoints[i-1].z),
                (waypoints[i].x, waypoints[i].y, waypoints[i].z)
            )
            distances.append(dist)
            total_distance += dist
        
        # Distribute points proportionally based on segment lengths
        cumulative_points = [0]
        for dist in distances:
            cumulative_points.append(cumulative_points[-1] + int(num_points * (dist / total_distance)))
        cumulative_points[-1] = num_points - 1  # Ensure last point is the total number of points
        
        # Interpolate each segment
        for segment in range(len(waypoints) - 1):
            start_wp = waypoints[segment]
            end_wp = waypoints[segment + 1]
            
            start_time = time_window.start + total_time * (cumulative_points[segment] / (num_points - 1))
            end_time = time_window.start + total_time * (cumulative_points[segment + 1] / (num_points - 1))
            
            # Generate points for this segment
            segment_points = cumulative_points[segment + 1] - cumulative_points[segment] + 1
            
            for j in range(segment_points):
                # Linear interpolation
                alpha = j / (segment_points - 1)
                
                x = start_wp.x + alpha * (end_wp.x - start_wp.x)
                y = start_wp.y + alpha * (end_wp.y - start_wp.y)
                z = start_wp.z + alpha * (end_wp.z - start_wp.z)
                
                # Interpolate time
                time = start_time + alpha * (end_time - start_time)
                
                state = DroneState(x, y, z, time)
                trajectory.append(state)
        
        return trajectory

    def _check_spatial_conflict(self, pos1: Tuple[float, float, float], 
                               pos2: Tuple[float, float, float]) -> bool:
        """
        Check if two positions conflict spatially based on the safety buffer.
        
        Args:
            pos1: (x, y, z) coordinates of first position
            pos2: (x, y, z) coordinates of second position
            
        Returns:
            True if there's a spatial conflict, False otherwise
        """
        dist = distance.euclidean(pos1, pos2)
        return dist < self.safety_buffer
    
    def _find_conflicts_on_full_paths(self, primary_trajectory: List[DroneState]) -> List[Dict]:
        """
        Find conflicts between the primary trajectory and all simulated flights
        using continuous path interpolation.
        
        Args:
            primary_trajectory: List of drone states for the primary mission
            
        Returns:
            List of conflict details
        """
        conflicts = []
        
        # Get full time range from primary trajectory
        min_time = primary_trajectory[0].time
        max_time = primary_trajectory[-1].time
        
        # Create interpolation functions for all flights
        interpolated_flights = []
        
        for simulated_flight in self.simulated_flights:
            if len(simulated_flight.states) < 2:
                continue
                
            # Sort flight states by time to ensure correct interpolation
            sorted_states = sorted(simulated_flight.states, key=lambda s: s.time)
            
            # Extract coordinates and times
            times = [state.time for state in sorted_states]
            xs = [state.x for state in sorted_states]
            ys = [state.y for state in sorted_states]
            zs = [state.z for state in sorted_states]
            
            # Check if flight has any overlap with primary mission time window
            if times[-1] < min_time or times[0] > max_time:
                continue
            
            # Limit interp domain to values within the flight time range
            flight_min_time = max(min_time, times[0])
            flight_max_time = min(max_time, times[-1])
            
            try:
                # Create interpolation functions for this flight
                fx = interp1d(times, xs, bounds_error=False, fill_value="extrapolate")
                fy = interp1d(times, ys, bounds_error=False, fill_value="extrapolate")
                fz = interp1d(times, zs, bounds_error=False, fill_value="extrapolate")
                
                interpolated_flights.append({
                    'drone_id': simulated_flight.drone_id,
                    'time_range': (flight_min_time, flight_max_time),
                    'interp_funcs': (fx, fy, fz)
                })
            except ValueError as e:
                print(f"Warning: Could not interpolate flight {simulated_flight.drone_id}: {e}")
        
        # Check for conflicts at each point in the primary trajectory
        for state in primary_trajectory:
            current_time = state.time
            primary_pos = (state.x, state.y, state.z)
            
            for flight in interpolated_flights:
                min_time, max_time = flight['time_range']
                
                # Skip if current time is outside flight's time range
                if current_time < min_time or current_time > max_time:
                    continue
                
                # Get interpolated position of this flight at current time
                fx, fy, fz = flight['interp_funcs']
                flight_x = float(fx(current_time))
                flight_y = float(fy(current_time))
                flight_z = float(fz(current_time))
                flight_pos = (flight_x, flight_y, flight_z)
                
                # Check for conflict
                if self._check_spatial_conflict(primary_pos, flight_pos):
                    conflicts.append({
                        'drone_id': flight['drone_id'],
                        'time': current_time,
                        'primary_position': primary_pos,
                        'conflict_position': flight_pos,
                        'distance': distance.euclidean(primary_pos, flight_pos)
                    })
        
        return conflicts
    
    def check_mission_safety(self, mission: WaypointMission) -> Dict:
        """
        Check if a mission is safe to execute.
        
        Args:
            mission: The waypoint mission to check
            
        Returns:
            A dictionary with safety status and conflict details if any
        """
        # Interpolate primary drone trajectory
        primary_trajectory = self._interpolate_trajectory(
            mission.waypoints,
            mission.time_window
        )
        
        # Find conflicts using full path interpolation
        conflicts = self._find_conflicts_on_full_paths(primary_trajectory)
        
        # Prepare result
        if not conflicts:
            return {
                'status': 'clear',
                'message': 'No conflicts detected. Mission is safe to execute.'
            }
        else:
            # Group conflicts by drone_id
            grouped_conflicts = {}
            for conflict in conflicts:
                drone_id = conflict['drone_id']
                if drone_id not in grouped_conflicts:
                    grouped_conflicts[drone_id] = []
                grouped_conflicts[drone_id].append(conflict)
            
            return {
                'status': 'conflict detected',
                'message': f'Found {len(conflicts)} potential conflicts with {len(grouped_conflicts)} drones.',
                'conflicts': conflicts,
                'grouped_conflicts': grouped_conflicts
            }
    
    def visualize_mission(self, mission: WaypointMission, 
                      result: Dict, show_3d: bool = True, 
                      save_path: Optional[str] = None) -> None:

        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[8, 1])
        ax = fig.add_subplot(gs[0], projection='3d')

        # Plot primary mission waypoints
        x_coords = [wp.x for wp in mission.waypoints]
        y_coords = [wp.y for wp in mission.waypoints]
        z_coords = [wp.z for wp in mission.waypoints]
        ax.plot(x_coords, y_coords, z_coords, 'bo-', markersize=8, linewidth=2, label='Primary Mission')
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, marker='^', label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, marker='s', label='End')

        # Interpolate primary trajectory
        primary_trajectory = self._interpolate_trajectory(mission.waypoints, mission.time_window)
        primary_x = [s.x for s in primary_trajectory]
        primary_y = [s.y for s in primary_trajectory]
        primary_z = [s.z for s in primary_trajectory]
        ax.plot(primary_x, primary_y, primary_z, 'b--', alpha=0.5, linewidth=1)

        # Plot simulated flights
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulated_flights)))
        for i, flight in enumerate(self.simulated_flights):
            x_sim = [s.x for s in flight.states]
            y_sim = [s.y for s in flight.states]
            z_sim = [s.z for s in flight.states]
            ax.plot(x_sim, y_sim, z_sim, '-', color=colors[i], linewidth=1, alpha=0.7, label=f'Drone {flight.drone_id}')

        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')
        ax.set_title('3D Visualization of Drone Mission and Conflicts', pad=50)
        ax.grid(True)
        ax.legend(loc='upper right')

        # Set equal bounds
        max_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)) / 2.0
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        ax.set_xlim(mid_x - max_range*1.2, mid_x + max_range*1.2)
        ax.set_ylim(mid_y - max_range*1.2, mid_y + max_range*1.2)
        ax.set_zlim(mid_z - max_range*1.2, mid_z + max_range*1.2)

        # Frame slider
        slider_ax = fig.add_subplot(gs[1])
        frame_slider = plt.Slider(slider_ax, 'Frame', 0, len(primary_trajectory)-1, valinit=0, valstep=1, color='lightgray')

        frame_point = ax.scatter([], [], [], color='purple', s=100, label='Primary Drone Position')
        sim_frame_points = []
        for i, flight in enumerate(self.simulated_flights):
            dot = ax.scatter([], [], [], color=colors[i], s=50, label=f'Drone {flight.drone_id} Position')
            sim_frame_points.append(dot)

        def update_frame(val):
            frame = int(frame_slider.val)
            current_time = primary_trajectory[frame].time

            # Update primary drone position
            frame_point._offsets3d = ([primary_x[frame]], [primary_y[frame]], [primary_z[frame]])

            # Update simulated positions
            for i, flight in enumerate(self.simulated_flights):
                closest_state = min(flight.states, key=lambda s: abs(s.time - current_time))
                sim_frame_points[i]._offsets3d = ([closest_state.x], [closest_state.y], [closest_state.z])

            # Remove previous dynamic conflict visuals
            for coll in ax.collections[:]:
                if hasattr(coll, '_is_dynamic_conflict'):
                    coll.remove()
            if hasattr(ax, '_safety_spheres'):
                for surf in ax._safety_spheres:
                    try:
                        surf.remove()
                    except:
                        pass
                ax._safety_spheres.clear()
            else:
                ax._safety_spheres = []

            # Show only current time conflicts
            conflict_tolerance = 1.0
            if result['status'] == 'conflict detected':
                for conflict in result['conflicts']:
                    if abs(conflict['time'] - current_time) <= conflict_tolerance:
                        x, y, z = conflict['primary_position']
                        marker = ax.scatter([x], [y], [z], color='red', s=150, alpha=0.8, marker='*')
                        marker._is_dynamic_conflict = True

                        # Safety sphere
                        u = np.linspace(0, 2 * np.pi, 10)
                        v = np.linspace(0, np.pi, 10)
                        sphere_x = x + self.safety_buffer * np.outer(np.cos(u), np.sin(v))
                        sphere_y = y + self.safety_buffer * np.outer(np.sin(u), np.sin(v))
                        sphere_z = z + self.safety_buffer * np.outer(np.ones(np.size(u)), np.cos(v))
                        sphere = ax.plot_surface(sphere_x, sphere_y, sphere_z, color='red', alpha=0.1)
                        ax._safety_spheres.append(sphere)

            state = primary_trajectory[frame]
            ax.set_title(f'3D Visualization - Frame {frame}\n'
                        f'Position: ({state.x:.2f}, {state.y:.2f}, {state.z:.2f}) m\n'
                        f'Time: {state.time:.2f} s')

            fig.canvas.draw_idle()

        frame_slider.on_changed(update_frame)

        # Initialize first frame
        frame_point._offsets3d = ([primary_x[0]], [primary_y[0]], [primary_z[0]])
        for i, flight in enumerate(self.simulated_flights):
            first_state = flight.states[0]
            sim_frame_points[i]._offsets3d = ([first_state.x], [first_state.y], [first_state.z])

        ax.legend(loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()
    
    def create_animation(self, mission: WaypointMission, save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create a 3D animation of the mission execution over time.
        
        Args:
            mission: The waypoint mission
            save_path: Path to save the animation as a video file
            
        Returns:
            The animation object
        """
        # Interpolate primary trajectory
        primary_trajectory = self._interpolate_trajectory(
            mission.waypoints,
            mission.time_window,
            num_points=200
        )
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the time range for animation
        min_time = mission.time_window.start
        max_time = mission.time_window.end
        
        # Prepare static elements
        # Plot all simulated flight paths (static)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulated_flights)))
        for i, flight in enumerate(self.simulated_flights):
            x_sim = [state.x for state in flight.states]
            y_sim = [state.y for state in flight.states]
            z_sim = [state.z for state in flight.states]
            ax.plot(x_sim, y_sim, z_sim, '-', color=colors[i], linewidth=1, alpha=0.3, 
                    label=f'Drone {flight.drone_id} Path')
        
        # Plot primary mission waypoints (static)
        x_coords = [wp.x for wp in mission.waypoints]
        y_coords = [wp.y for wp in mission.waypoints]
        z_coords = [wp.z for wp in mission.waypoints]
        ax.plot(x_coords, y_coords, z_coords, 'bo-', markersize=6, linewidth=1, alpha=0.5, label='Primary Mission Path')
        
        # Dynamic elements to be updated in animation
        # We need to use a ScatterPlot for 3D animation instead of plot
        primary_dot = ax.scatter([], [], [], color='blue', s=80, label='Primary Drone')
        
        sim_dots = []
        for i in range(len(self.simulated_flights)):
            dot = ax.scatter([], [], [], color=colors[i], s=50)
            sim_dots.append(dot)
        
        # Buffer sphere (for conflict visualization)
        # We'll create and update this sphere in the animation function
        
        # Conflict markers
        conflict_markers = ax.scatter([], [], [], color='red', s=120, marker='*', label='Conflict')
        
        # Time display
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Setup plot
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')
        ax.set_title('3D Drone Mission Execution Animation')
        
        # Get plot bounds
        all_x = x_coords.copy()
        all_y = y_coords.copy()
        all_z = z_coords.copy()
        for flight in self.simulated_flights:
            all_x.extend([state.x for state in flight.states])
            all_y.extend([state.y for state in flight.states])
            all_z.extend([state.z for state in flight.states])
        
        # Set equal aspect ratio by calculating the bounds
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        z_range = max(all_z) - min(all_z)
        max_range = max(x_range, y_range, z_range) / 2.0
        
        mid_x = (max(all_x) + min(all_x)) * 0.5
        mid_y = (max(all_y) + min(all_y)) * 0.5
        mid_z = (max(all_z) + min(all_z)) * 0.5
        
        ax.set_xlim(mid_x - max_range*1.2, mid_x + max_range*1.2)
        ax.set_ylim(mid_y - max_range*1.2, mid_y + max_range*1.2)
        ax.set_zlim(mid_z - max_range*1.2, mid_z + max_range*1.2)
        
        ax.grid(True)
        plt.legend(loc='upper right')
        
        # Helper function to create a safety buffer sphere
        def create_sphere(x, y, z, radius):
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            
            x_sphere = x + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = y + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            return x_sphere, y_sphere, z_sphere
        
        # Variable to store the current buffer sphere
        buffer_sphere = None
        
        def init():
            primary_dot._offsets3d = (np.array([]), np.array([]), np.array([]))
            
            for dot in sim_dots:
                dot._offsets3d = (np.array([]), np.array([]), np.array([]))
                
            conflict_markers._offsets3d = (np.array([]), np.array([]), np.array([]))
            time_text.set_text('')
            
            # Initialize with all artists to be returned
            artists = [primary_dot, conflict_markers, time_text] + sim_dots
            
            nonlocal buffer_sphere
            # Add an initial empty buffer sphere (will be updated later)
            if buffer_sphere is None:
                x_sphere, y_sphere, z_sphere = create_sphere(0, 0, 0, self.safety_buffer)
                buffer_sphere = ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                                               color='blue', alpha=0.1, visible=False)
            else:
                buffer_sphere.set_visible(False)
                
            artists.append(buffer_sphere)
            return artists
        
        def get_drone_positions_at_time(t):
            # Get primary drone position
            primary_pos = None
            for state in primary_trajectory:
                if abs(state.time - t) < 0.1:  # Close enough to the target time
                    primary_pos = (state.x, state.y, state.z)
                    break
            
            # If no exact match, interpolate between closest states
            if primary_pos is None and len(primary_trajectory) > 1:
                # Find the two surrounding states
                for i in range(1, len(primary_trajectory)):
                    if primary_trajectory[i-1].time <= t <= primary_trajectory[i].time:
                        t1, t2 = primary_trajectory[i-1].time, primary_trajectory[i].time
                        x1, x2 = primary_trajectory[i-1].x, primary_trajectory[i].x
                        y1, y2 = primary_trajectory[i-1].y, primary_trajectory[i].y
                        z1, z2 = primary_trajectory[i-1].z, primary_trajectory[i].z
                        
                        # Linear interpolation
                        if t2 != t1:  # Avoid division by zero
                            ratio = (t - t1) / (t2 - t1)
                            x = x1 + ratio * (x2 - x1)
                            y = y1 + ratio * (y2 - y1)
                            z = z1 + ratio * (z2 - z1)
                            primary_pos = (x, y, z)
                        else:
                            primary_pos = (x1, y1, z1)
                        break
            
            # Get simulated drone positions
            sim_positions = []
            for flight in self.simulated_flights:
                pos = None
                for state in flight.states:
                    if abs(state.time - t) < 0.1:
                        pos = (state.x, state.y, state.z)
                        break
                
                # If no exact match, interpolate
                if pos is None and len(flight.states) > 1:
                    for i in range(1, len(flight.states)):
                        if flight.states[i-1].time <= t <= flight.states[i].time:
                            t1, t2 = flight.states[i-1].time, flight.states[i].time
                            x1, x2 = flight.states[i-1].x, flight.states[i].x
                            y1, y2 = flight.states[i-1].y, flight.states[i].y
                            z1, z2 = flight.states[i-1].z, flight.states[i].z
                            
                            if t2 != t1:
                                ratio = (t - t1) / (t2 - t1)
                                x = x1 + ratio * (x2 - x1)
                                y = y1 + ratio * (y2 - y1)
                                z = z1 + ratio * (z2 - z1)
                                pos = (x, y, z)
                            else:
                                pos = (x1, y1, z1)
                            break
                
                sim_positions.append(pos)
            
            return primary_pos, sim_positions
        
        def animate(frame):
            nonlocal buffer_sphere
            
            # Remove the previous buffer sphere
            if buffer_sphere is not None:
                buffer_sphere.remove()
                buffer_sphere = None
                
            # Map frame to time
            t = min_time + (max_time - min_time) * (frame / 100)
            
            # Get positions at this time
            primary_pos, sim_positions = get_drone_positions_at_time(t)
            
            # Update primary drone position and buffer sphere
            if primary_pos:
                x, y, z = primary_pos
                primary_dot._offsets3d = ([x], [y], [z])
                
                # Create and add a new buffer sphere
                x_sphere, y_sphere, z_sphere = create_sphere(x, y, z, self.safety_buffer)
                buffer_sphere = ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                                              color='blue', alpha=0.1)
            else:
                primary_dot._offsets3d = (np.array([]), np.array([]), np.array([]))
            
            # Update simulated drone positions
            conflicts = []
            for i, pos in enumerate(sim_positions):
                if pos:
                    x, y, z = pos
                    sim_dots[i]._offsets3d = ([x], [y], [z])
                    
                    # Check for conflict
                    if primary_pos and self._check_spatial_conflict(primary_pos, pos):
                        conflicts.append(pos)
                else:
                    sim_dots[i]._offsets3d = (np.array([]), np.array([]), np.array([]))
            
            # Update conflict markers
            if conflicts:
                x_conflicts = [pos[0] for pos in conflicts]
                y_conflicts = [pos[1] for pos in conflicts]
                z_conflicts = [pos[2] for pos in conflicts]
                conflict_markers._offsets3d = (x_conflicts, y_conflicts, z_conflicts)
            else:
                conflict_markers._offsets3d = (np.array([]), np.array([]), np.array([]))
            
            # Update time display
            hours = int(t // 3600)
            minutes = int((t % 3600) // 60)
            seconds = int(t % 60)
            time_text.set_text(f'Time: {hours:02d}:{minutes:02d}:{seconds:02d}')
            
            # Return all artists that were updated
            artists = [primary_dot, conflict_markers, time_text, buffer_sphere] + sim_dots
            return artists
        
        # Create animation
        ani = FuncAnimation(fig, animate, frames=101, init_func=init, 
                           interval=50, blit=False)  # Changed to blit=False for 3D
        
        # Save animation if path provided
        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='ffmpeg', fps=20, dpi=200)
        
        return ani


def generate_test_data():
    """
    Generate test data for demonstration.
    
    Returns:
        A tuple containing primary mission and simulated flights
    """
    # Create a primary mission
    primary_waypoints = [
        Waypoint(x=0, y=0, z=10),
        Waypoint(x=50, y=30, z=20),
        Waypoint(x=100, y=0, z=15),
        Waypoint(x=150, y=50, z=25),
        Waypoint(x=200, y=0, z=10)
    ]
    
    primary_time_window = TimeWindow(start=0, end=600)  # 10 minutes mission
    
    primary_mission = WaypointMission(
        waypoints=primary_waypoints,
        time_window=primary_time_window
    )
    
    # Create simulated flights
    simulated_flights = []
    
    # Simulated Flight 1 - No conflict
    states_1 = []
    for i in range(11):
        t = i * 60  # Every minute
        states_1.append(DroneState(
            x=75 + i * 5,
            y=-50 - i * 3,
            z=30,
            time=t
        ))
    
    simulated_flights.append(SimulatedFlight(
        drone_id="Drone-001",
        states=states_1
    ))
    
    # Simulated Flight 2 - Potential conflict
    states_2 = []
    for i in range(11):
        t = i * 60  # Every minute
        states_2.append(DroneState(
            x=40 + i * 10,
            y=20 + i * 3,
            z=20,
            time=t
        ))
    
    simulated_flights.append(SimulatedFlight(
        drone_id="Drone-002",
        states=states_2
    ))
    
    # Simulated Flight 3 - Another potential conflict
    states_3 = []
    for i in range(11):
        t = i * 60  # Every minute
        states_3.append(DroneState(
            x=190 - i * 15,
            y=10 + i * 5,
            z=15,
            time=t
        ))
    
    simulated_flights.append(SimulatedFlight(
        drone_id="Drone-003",
        states=states_3
    ))

    # Simulated Flight 4 - Another potential conflict
    states_4 = []
    for i in range(11):
        t = i * 60  # Every minute
        states_4.append(DroneState(
            x=150 - i * 5,
            y=10 + i * 2,
            z=7,
            time=t
        ))
    
    simulated_flights.append(SimulatedFlight(
        drone_id="Drone-004",
        states=states_4
    ))
    
    return primary_mission, simulated_flights

def record_primary_mission():
    """
    Collect primary mission data from user input.

    Returns:
        WaypointMission: A new mission created from user-provided waypoints and time window.
    """
    print("\n--- Record Primary Mission ---")
    
    waypoints = []
    num_waypoints = int(input("Enter number of waypoints: "))
    
    for i in range(num_waypoints):
        print(f"\nWaypoint {i+1}:")
        x = float(input("  Enter x coordinate (meters): "))
        y = float(input("  Enter y coordinate (meters): "))
        z = float(input("  Enter altitude z (meters): "))
        waypoints.append(Waypoint(x=x, y=y, z=z))
    
    print("\nEnter mission time window:")
    start = int(input("  Start time (seconds from t=0): "))
    end = int(input("  End time (seconds from t=0): "))
    time_window = TimeWindow(start=start, end=end)

    mission = WaypointMission(waypoints=waypoints, time_window=time_window)
    
    print("\nPrimary mission recorded successfully!")
    return mission


def demo():
    """Run a demonstration of the deconfliction system."""
    print("-" * 50)
    print("UAV Strategic Deconfliction System Demonstration")
    print("-" * 50)

    # Initialize the system
    system = DeconflictionSystem(safety_buffer=15.0)
    
    mission, simulated_flights = generate_test_data()

    # Let the user choose mission type
    print("\nChoose mission type:")
    print("1. Use default primary mission")
    print("2. Record a new primary mission")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        mission = record_primary_mission()
    
    system.register_simulated_flights(simulated_flights)
    
    result = system.check_mission_safety(mission)
    
    print(f"Safety check result: {result['status']}")
    print(result['message'])
    
    if result['status'] == 'conflict detected':
        print("\nConflict Details:")
        for drone_id, conflicts in result['grouped_conflicts'].items():
            print(f"\n  Drone {drone_id}:")
            for i, conflict in enumerate(conflicts[:3]):  # Show at most 3 conflicts per drone
                time_str = f"{int(conflict['time'] // 60):02d}:{int(conflict['time'] % 60):02d}"
                print(f"    Conflict {i+1} at time {time_str}:")
                print(f"      Primary drone at: {conflict['primary_position']}")
                print(f"      {drone_id} at: {conflict['conflict_position']}")
                print(f"      Distance: {conflict['distance']:.2f}m (Safety buffer: {system.safety_buffer}m)")
            
            # If there are more conflicts than we displayed
            if len(conflicts) > 3:
                print(f"    ... and {len(conflicts) - 3} more conflicts")
    
    # Visualize
    print("\nGenerating visualization...")
    system.visualize_mission(mission, result, show_3d=True)
    
    # Create animation
    print("\nGenerating animation...")
    ani = system.create_animation(mission, save_path="mission_animation.mp4")
    plt.show()


if __name__ == "__main__":
    demo()