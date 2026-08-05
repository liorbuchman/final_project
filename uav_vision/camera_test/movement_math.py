def calculate_movement(current_angle, mic_target_doa, min_limit=-175.0, max_limit=175.0):
    """
    Calculates the required movement to align the camera with the target DOA."""
  
    target_angle = mic_target_doa
    if target_angle > 180:
        target_angle = target_angle - 360  # למשל: 270 הופך ל-90- (שמאלה)
        
    safe_target = max(min_limit, min(target_angle, max_limit))
    
    if target_angle != safe_target:
        print(f"[Warning] Target {target_angle} is in the dead zone. Clamped to {safe_target}.")
    
    diff = safe_target - current_angle
    
    # 4. כיוון וכמות מעלות
    direction = "Right" if diff > 0 else "Left" if diff < 0 else "None"
    degrees_to_move = abs(diff)
    
    return direction, safe_target, degrees_to_move