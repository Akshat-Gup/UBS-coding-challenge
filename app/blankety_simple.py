"""
Lightweight version of blankety for reliable deployment.
Simple linear interpolation without heavy scipy dependencies.
"""

def simple_linear_interpolation(series):
    """Simple linear interpolation for missing values."""
    if not series:
        return []
    
    # Convert to list and handle None values
    values = []
    indices = []
    
    for i, val in enumerate(series):
        if val is not None:
            values.append(float(val))
            indices.append(i)
    
    if len(values) == 0:
        return [0.0] * len(series)
    
    if len(values) == 1:
        return [values[0]] * len(series)
    
    # Simple linear interpolation
    result = [0.0] * len(series)
    
    for i in range(len(series)):
        if series[i] is not None:
            result[i] = float(series[i])
        else:
            # Find surrounding values
            left_idx = None
            right_idx = None
            
            # Find left value
            for j in range(i-1, -1, -1):
                if series[j] is not None:
                    left_idx = j
                    break
            
            # Find right value
            for j in range(i+1, len(series)):
                if series[j] is not None:
                    right_idx = j
                    break
            
            # Interpolate
            if left_idx is not None and right_idx is not None:
                left_val = float(series[left_idx])
                right_val = float(series[right_idx])
                ratio = (i - left_idx) / (right_idx - left_idx)
                result[i] = left_val + ratio * (right_val - left_val)
            elif left_idx is not None:
                result[i] = float(series[left_idx])
            elif right_idx is not None:
                result[i] = float(series[right_idx])
            else:
                result[i] = 0.0
    
    return result


def blankety_blanks_simple(payload):
    """
    Simple version of blankety_blanks that works without heavy dependencies.
    
    Expected payload:
    {
        "series": [
            [0.12, 0.17, null, 0.30, 0.29, null, ...],
            [1.02, null, 0.98, 0.97, null, ...],
            ...
        ]
    }
    
    Returns:
    {
        "answer": [
            [0.12, 0.17, 0.23, 0.30, 0.29, 0.31, ...],
            [1.02, 1.01, 0.98, 0.97, 0.96, ...],
            ...
        ]
    }
    """
    try:
        input_series = payload.get("series", [])
        
        if not input_series:
            return {"answer": []}
        
        # Process each series
        imputed_series = []
        for series in input_series:
            if not isinstance(series, list):
                # Handle malformed input
                imputed_series.append([0.0] * 1000)
                continue
                
            # Ensure the series has exactly 1000 elements
            if len(series) != 1000:
                # Pad or truncate to 1000 elements
                if len(series) < 1000:
                    series.extend([None] * (1000 - len(series)))
                else:
                    series = series[:1000]
            
            # Impute missing values using simple linear interpolation
            imputed = simple_linear_interpolation(series)
            
            # Ensure exactly 1000 elements in output
            if len(imputed) != 1000:
                if len(imputed) < 1000:
                    # Pad with last value
                    pad_value = imputed[-1] if imputed else 0.0
                    imputed.extend([pad_value] * (1000 - len(imputed)))
                else:
                    imputed = imputed[:1000]
            
            imputed_series.append(imputed)
        
        # Ensure we have exactly 100 series
        while len(imputed_series) < 100:
            imputed_series.append([0.0] * 1000)
        
        if len(imputed_series) > 100:
            imputed_series = imputed_series[:100]
        
        return {"answer": imputed_series}
    
    except Exception as e:
        # Return safe fallback in case of any error
        return {"answer": [[0.0] * 1000 for _ in range(100)]}


# Test function
def test_simple_imputation():
    """Test the simple imputation with sample data."""
    test_data = {
        "series": [
            [0.10, None, 0.30, None, 0.52] + [0.1 * i for i in range(995)],
            [1.02, None, 0.98, 0.97, None] + [1.0 + 0.01 * i for i in range(995)]
        ]
    }
    
    # Pad to make it 100 series of 1000 elements each
    for i in range(98):
        series = [float(i + j * 0.001) if j % 10 != 3 else None for j in range(1000)]
        test_data["series"].append(series)
    
    result = blankety_blanks_simple(test_data)
    print(f"Processed {len(result['answer'])} series")
    print(f"Each series has {len(result['answer'][0])} elements")
    print(f"First series first 10 elements: {result['answer'][0][:10]}")
    
    # Check for null values
    has_nulls = any(
        any(x is None for x in series) 
        for series in result['answer']
    )
    print(f"Contains null values: {has_nulls}")


if __name__ == "__main__":
    test_simple_imputation()
