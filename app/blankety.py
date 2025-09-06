import numpy as np
from typing import List, Dict, Any, Optional
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')


def detect_pattern_type(series: List[float]) -> str:
    """Detect the type of pattern in the series to choose the best imputation method."""
    # Convert to numpy array, handling None values
    clean_values = [x for x in series if x is not None]
    if len(clean_values) < 3:
        return "linear"
    
    arr = np.array(clean_values)
    indices = [i for i, x in enumerate(series) if x is not None]
    
    # Check for trend
    try:
        correlation = np.corrcoef(indices, arr)[0, 1]
        if abs(correlation) > 0.7:
            return "trend"
    except:
        pass
    
    # Check for periodicity (simple autocorrelation)
    if len(arr) > 10:
        try:
            # Look for periodic patterns
            autocorr = np.correlate(arr, arr, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            if len(autocorr) > 5 and max(autocorr[2:min(10, len(autocorr))]) > 0.8 * autocorr[0]:
                return "periodic"
        except:
            pass
    
    return "smooth"


def impute_linear(series: List[Optional[float]]) -> List[float]:
    """Simple linear interpolation."""
    arr = np.array(series, dtype=float)
    valid_mask = ~np.isnan(arr)
    
    if np.sum(valid_mask) < 2:
        # Not enough valid points, use mean or forward fill
        mean_val = np.nanmean(arr)
        if np.isnan(mean_val):
            mean_val = 0.0
        return [mean_val if x is None else x for x in series]
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_mask]
    
    # Interpolate missing values
    f = interp1d(valid_indices, valid_values, kind='linear', 
                 bounds_error=False, fill_value='extrapolate')
    
    result = f(np.arange(len(series)))
    return result.tolist()


def impute_spline(series: List[Optional[float]], smoothing: float = 0.1) -> List[float]:
    """Spline interpolation with smoothing."""
    arr = np.array(series, dtype=float)
    valid_mask = ~np.isnan(arr)
    
    if np.sum(valid_mask) < 4:
        return impute_linear(series)
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_mask]
    
    try:
        # Use univariate spline with smoothing
        spline = UnivariateSpline(valid_indices, valid_values, s=smoothing * len(valid_values))
        result = spline(np.arange(len(series)))
        return result.tolist()
    except:
        return impute_linear(series)


def impute_trend(series: List[Optional[float]]) -> List[float]:
    """Impute using polynomial trend."""
    arr = np.array(series, dtype=float)
    valid_mask = ~np.isnan(arr)
    
    if np.sum(valid_mask) < 3:
        return impute_linear(series)
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_mask]
    
    try:
        # Fit polynomial (degree 2 for quadratic trend)
        degree = min(2, len(valid_values) - 1)
        coeffs = np.polyfit(valid_indices, valid_values, degree)
        poly = np.poly1d(coeffs)
        result = poly(np.arange(len(series)))
        return result.tolist()
    except:
        return impute_linear(series)


def impute_savgol(series: List[Optional[float]]) -> List[float]:
    """Impute using Savitzky-Golay filter for smooth trends."""
    # First do linear interpolation to fill gaps
    linear_result = impute_linear(series)
    
    try:
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(21, len(linear_result) // 4 * 2 + 1)  # Ensure odd number
        window_length = max(5, window_length)
        
        if window_length >= len(linear_result):
            return linear_result
            
        smoothed = savgol_filter(linear_result, window_length, 3)
        return smoothed.tolist()
    except:
        return linear_result


def impute_local_regression(series: List[Optional[float]], window_size: int = 20) -> List[float]:
    """Local regression imputation."""
    arr = np.array(series, dtype=float)
    result = arr.copy()
    
    missing_indices = np.where(np.isnan(arr))[0]
    
    for idx in missing_indices:
        # Define local window
        start = max(0, idx - window_size // 2)
        end = min(len(arr), idx + window_size // 2 + 1)
        
        local_series = arr[start:end]
        local_indices = np.arange(start, end)
        valid_mask = ~np.isnan(local_series)
        
        if np.sum(valid_mask) >= 2:
            try:
                # Local linear regression
                valid_local_indices = local_indices[valid_mask]
                valid_local_values = local_series[valid_mask]
                
                coeffs = np.polyfit(valid_local_indices, valid_local_values, 1)
                result[idx] = np.polyval(coeffs, idx)
            except:
                # Fallback to local mean
                result[idx] = np.nanmean(local_series)
        else:
            # Fallback to global mean
            result[idx] = np.nanmean(arr)
    
    return result.tolist()


def impute_series(series: List[Optional[float]]) -> List[float]:
    """Main imputation function that chooses the best method based on pattern detection."""
    # Handle edge cases
    if not series:
        return []
    
    # Convert None to NaN for processing
    processed_series = [float('nan') if x is None else float(x) for x in series]
    
    # If no missing values, return as is
    if not any(np.isnan(processed_series)):
        return processed_series
    
    # If all values are missing, return zeros
    if all(np.isnan(processed_series)):
        return [0.0] * len(series)
    
    # Detect pattern and choose method
    pattern_type = detect_pattern_type(series)
    
    try:
        if pattern_type == "trend":
            result = impute_trend(processed_series)
        elif pattern_type == "periodic":
            result = impute_spline(processed_series, smoothing=0.05)
        elif pattern_type == "smooth":
            result = impute_savgol(processed_series)
        else:
            result = impute_local_regression(processed_series)
        
        # Ensure no NaN or inf values in result
        result = [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in result]
        
        return result
    
    except Exception:
        # Ultimate fallback: linear interpolation
        try:
            result = impute_linear(processed_series)
            result = [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in result]
            return result
        except:
            # Final fallback: forward fill with mean
            mean_val = np.nanmean([x for x in processed_series if not np.isnan(x)])
            if np.isnan(mean_val):
                mean_val = 0.0
            return [mean_val] * len(series)


def blankety_blanks(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to impute missing values in time series data.
    
    Expected payload structure:
    {
        "series": [
            [0.12, 0.17, null, 0.30, 0.29, null, ...],  # 1000 elements
            [1.02, null, 0.98, 0.97, null, ...],        # 1000 elements
            ...  # 100 total series
        ]
    }
    
    Returns:
    {
        "answer": [
            [0.12, 0.17, 0.23, 0.30, 0.29, 0.31, ...],  # 1000 elements, no nulls
            [1.02, 1.01, 0.98, 0.97, 0.96, ...],        # 1000 elements, no nulls
            ...  # 100 total series
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
            
            # Impute missing values
            imputed = impute_series(series)
            
            # Ensure exactly 1000 elements in output
            if len(imputed) != 1000:
                if len(imputed) < 1000:
                    # Pad with last value or mean
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


# Test function for development
def test_imputation():
    """Test the imputation with sample data."""
    # Create test data with missing values
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
    
    result = blankety_blanks(test_data)
    print(f"Processed {len(result['answer'])} series")
    print(f"Each series has {len(result['answer'][0])} elements")
    print(f"First series first 10 elements: {result['answer'][0][:10]}")
    
    # Check for null values
    has_nulls = any(
        any(x is None or np.isnan(x) for x in series) 
        for series in result['answer']
    )
    print(f"Contains null values: {has_nulls}")


if __name__ == "__main__":
    test_imputation()
