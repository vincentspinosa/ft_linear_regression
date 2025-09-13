import os

def load_params(filename='thetas.txt') -> dict:
    params = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 6:
                params['theta0'] = float(lines[0].strip())
                params['theta1'] = float(lines[1].strip())
                params['min_km'] = float(lines[2].strip())
                params['max_km'] = float(lines[3].strip())
                params['min_price'] = float(lines[4].strip())
                params['max_price'] = float(lines[5].strip())
    return params

def normalize_value(value, min_val, max_val) -> float:
    return (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0

def denormalize_value(normalized_value, min_val, max_val) -> float:
    return normalized_value * (max_val - min_val) + min_val

def predict_price(normalized_mileage, theta0, theta1) -> float:
    return theta0 + (theta1 * normalized_mileage)

def main() -> None:
    params = load_params()

    if not params:
        print("Model not trained yet. Please run train.py first.")
        theta0, theta1 = 0.0, 0.0
        min_km, max_km = 0, 1
        min_price, max_price = 0, 1
    else:
        theta0 = params['theta0']
        theta1 = params['theta1']
        min_km = params['min_km']
        max_km = params['max_km']
        min_price = params['min_price']
        max_price = params['max_price']

    try:
        mileage = float(input("Enter the mileage of the car: "))
        normalized_mileage = normalize_value(mileage, min_km, max_km)
        normalized_price = predict_price(normalized_mileage, theta0, theta1)
        estimated_price = denormalize_value(normalized_price, min_price, max_price)
        print(f"The estimated price for a car with {mileage} km is: {estimated_price:.2f}")

    except ValueError:
        print("Invalid input. Please enter a valid number for mileage.")
    except KeyError:
        print("Error: Thetas file is incomplete or corrupted. Please retrain the model.")


if __name__ == "__main__":
    main()