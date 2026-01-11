#!/usr/bin/env python3
"""
Script to simplify URDF by removing collision geometry from decorative/small parts.
This reduces the collision pair count significantly for better PhysX performance.
"""

import re

# Parts to remove collision from (keep visual only)
PARTS_TO_SIMPLIFY = [
    'vis',  # All screws
    'vis_1',
    'vis_à_tête_tronconique',
    'composant2',  # Servo internal components
    'composant3',
    'composant4',
    'servo_motor_sts3215',  # Motor internals (keep holder collision)
    'servo_shield',  # Decorative shields
    'battery_holder',  # Non-functional for physics
    'back_middle_connector_panel',
    'intelrealsense',  # Camera (won't affect locomotion)
]

def should_remove_collision(part_name):
    """Check if collision should be removed for this part."""
    for pattern in PARTS_TO_SIMPLIFY:
        if pattern in part_name.lower():
            return True
    return False

def remove_collision_blocks(urdf_content):
    """Remove collision blocks while keeping visual blocks."""
    lines = urdf_content.split('\n')
    output_lines = []
    in_collision_block = False
    skip_collision = False
    current_part = None
    indent_level = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect part comment
        if '<!-- Part' in line:
            match = re.search(r'<!-- Part (.+?) -->', line)
            if match:
                current_part = match.group(1)
        
        # Check if we're entering a collision block
        if '<collision>' in line:
            in_collision_block = True
            indent_level = len(line) - len(line.lstrip())
            
            # Check if we should skip this collision block
            if current_part and should_remove_collision(current_part):
                skip_collision = True
                print(f"Removing collision for: {current_part}")
                i += 1
                continue
        
        # Skip lines in collision block if needed
        if in_collision_block and skip_collision:
            if '</collision>' in line:
                in_collision_block = False
                skip_collision = False
            i += 1
            continue
        
        # Keep the line
        output_lines.append(line)
        
        # Track when we exit collision block
        if in_collision_block and '</collision>' in line:
            in_collision_block = False
        
        i += 1
    
    return '\n'.join(output_lines)

def main():
    input_file = 'robot_simplified.urdf'
    output_file = 'robot_simplified.urdf'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        urdf_content = f.read()
    
    print("Removing collision geometry from decorative parts...")
    simplified_content = remove_collision_blocks(urdf_content)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(simplified_content)
    
    # Calculate size reduction
    original_size = len(urdf_content)
    new_size = len(simplified_content)
    reduction = ((original_size - new_size) / original_size) * 100
    
    print(f"\nDone!")
    print(f"Original size: {original_size:,} bytes")
    print(f"New size: {new_size:,} bytes")
    print(f"Reduction: {reduction:.1f}%")
    print(f"\nSimplified URDF saved to: {output_file}")
    print("\nParts with collision removed:")
    for part in PARTS_TO_SIMPLIFY:
        print(f"  - {part}*")

if __name__ == '__main__':
    main()
