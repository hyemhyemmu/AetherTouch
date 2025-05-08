import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class InteractionTracker:
    """
    Records and analyzes system interactions data
    
    This class provides functionality to log, analyze, and visualize
    interaction events between users and the Raspberry Pi system.
    """
    def __init__(self, log_dir="logs"):
        """
        Initialize the interaction tracker
        
        Args:
            log_dir (str): Directory to store log files
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique log file name based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"interaction_log_{timestamp}.csv")
        
        # Initialize start time
        self.start_time = time.time()
        
        # Initialize log file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "device", "gesture", "duration"])
        
        # Initialize counters for statistics
        self.stats = {
            "total_interactions": 0,
            "device_counts": {},
            "gesture_counts": {},
            "interaction_times": []
        }
        
        print(f"Interaction tracking started. Log file: {self.log_file}")
    
    def record_event(self, action, device, gesture="POINTING", duration=0):
        """
        Record an interaction event
        
        Args:
            action (str): Action performed (e.g., "ON", "OFF", "ADJUST")
            device (str): Device affected (e.g., "LED", "BUZZER")
            gesture (str): Gesture used for the interaction
            duration (float): Duration of interaction in seconds
        """
        timestamp = time.time() - self.start_time
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{timestamp:.2f}", action, device, gesture, f"{duration:.2f}"])
        
        # Update statistics
        self.stats["total_interactions"] += 1
        
        if device in self.stats["device_counts"]:
            self.stats["device_counts"][device] += 1
        else:
            self.stats["device_counts"][device] = 1
            
        if gesture in self.stats["gesture_counts"]:
            self.stats["gesture_counts"][gesture] += 1
        else:
            self.stats["gesture_counts"][gesture] = 1
            
        self.stats["interaction_times"].append(timestamp)
    
    def create_analysis_report(self, output_dir="reports"):
        """
        Generate comprehensive analysis report with visualizations
        
        Args:
            output_dir (str): Directory to save reports
            
        Returns:
            str: Path to generated report directory or None if error
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Read log data
        try:
            data = pd.read_csv(self.log_file)
        except Exception as e:
            print(f"Error reading log file: {e}")
            return None
        
        # Generate basic statistics
        if len(data) == 0:
            print("No interaction data to analyze")
            return None
            
        # 1. Device usage pie chart
        try:
            device_counts = data['device'].value_counts()
            plt.figure(figsize=(10, 6))
            plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Device Usage Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'device_usage.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating device usage chart: {e}")
        
        # 2. Gesture usage pie chart
        try:
            gesture_counts = data['gesture'].value_counts()
            plt.figure(figsize=(10, 6))
            plt.pie(gesture_counts, labels=gesture_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Gesture Usage Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'gesture_usage.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating gesture usage chart: {e}")
            
        # 3. Activity timeline
        try:
            plt.figure(figsize=(12, 6))
            
            # Create timeline with markers for each action
            action_colors = {
                'ON': 'go',    # Green circle for ON
                'OFF': 'ro',   # Red circle for OFF
                'ADJUST': 'bo' # Blue circle for ADJUST
            }
            
            for i, row in data.iterrows():
                marker = action_colors.get(row['action'], 'ko')  # Default to black
                plt.plot(row['timestamp'], 1, marker, markersize=10)
            
            # Add device labels
            for i, row in data.iterrows():
                plt.annotate(row['device'], 
                            (row['timestamp'], 1), 
                            textcoords="offset points",
                            xytext=(0,10), 
                            ha='center')
            
            plt.yticks([])  # Hide Y-axis ticks
            plt.title('Interaction Timeline')
            plt.xlabel('Time (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'activity_timeline.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating timeline: {e}")
            
        # 4. Interaction frequency over time (histogram)
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(data['timestamp'], bins=20)
            plt.title('Interaction Frequency Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Number of Interactions')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'interaction_frequency.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating frequency histogram: {e}")
        
        # 5. Generate text summary
        try:
            summary_file = os.path.join(report_dir, 'summary.txt')
            with open(summary_file, 'w') as f:
                f.write("# Interaction Analysis Summary\n\n")
                f.write(f"Total interactions: {len(data)}\n")
                f.write(f"Session duration: {data['timestamp'].max():.2f} seconds\n")
                f.write(f"Average interaction rate: {len(data)/data['timestamp'].max():.2f} per second\n\n")
                
                f.write("## Device Usage\n")
                for device, count in device_counts.items():
                    f.write(f"- {device}: {count} interactions ({count/len(data)*100:.1f}%)\n")
                
                f.write("\n## Gesture Usage\n")
                for gesture, count in gesture_counts.items():
                    f.write(f"- {gesture}: {count} interactions ({count/len(data)*100:.1f}%)\n")
                
                f.write("\n## Action Distribution\n")
                action_counts = data['action'].value_counts()
                for action, count in action_counts.items():
                    f.write(f"- {action}: {count} instances ({count/len(data)*100:.1f}%)\n")
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            
        print(f"Analysis report generated in: {report_dir}")
        return report_dir

    def export_data(self, format='csv'):
        """
        Export logged data to different formats
        
        Args:
            format (str): Export format ('csv', 'json', or 'excel')
            
        Returns:
            str: Path to exported file
        """
        data = pd.read_csv(self.log_file)
        export_dir = os.path.dirname(self.log_file)
        base_name = os.path.splitext(os.path.basename(self.log_file))[0]
        
        if format.lower() == 'json':
            output_path = os.path.join(export_dir, f"{base_name}.json")
            data.to_json(output_path, orient='records')
        elif format.lower() == 'excel':
            output_path = os.path.join(export_dir, f"{base_name}.xlsx")
            data.to_excel(output_path, index=False)
        else:  # Default to CSV
            output_path = self.log_file
            
        return output_path

# For backward compatibility
InteractionLogger = InteractionTracker
