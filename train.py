import csv
import os

def load_data(filename) -> tuple:
    mileage_data = []
    price_data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            mileage_data.append(float(row[0]))
            price_data.append(float(row[1]))
    return mileage_data, price_data

def normalize_data(data, min_val, max_val) -> list:
    return [(val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0 for val in data]

def estimate_price(normalized_mileage, theta0, theta1) -> float:
    return theta0 + (theta1 * normalized_mileage)

def train_model(normalized_mileage_data, normalized_price_data, learning_rate, iterations) -> tuple:
    theta0 = 0.0
    theta1 = 0.0
    m = len(normalized_mileage_data)

    for i in range(iterations):
        tmp_theta0 = 0.0
        tmp_theta1 = 0.0
        
        for j in range(m):
            estimated_price = estimate_price(normalized_mileage_data[j], theta0, theta1)
            tmp_theta0 += (estimated_price - normalized_price_data[j])
            tmp_theta1 += (estimated_price - normalized_price_data[j]) * normalized_mileage_data[j]
        
        tmp_theta0 = learning_rate * (1/m) * tmp_theta0
        tmp_theta1 = learning_rate * (1/m) * tmp_theta1
        
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        
    return theta0, theta1

def save_params(theta0, theta1, min_km, max_km, min_price, max_price, filename='thetas.txt') -> None:
    with open(filename, 'w') as f:
        f.write(f"{theta0}\n")
        f.write(f"{theta1}\n")
        f.write(f"{min_km}\n")
        f.write(f"{max_km}\n")
        f.write(f"{min_price}\n")
        f.write(f"{max_price}\n")

def main() -> None:
    data_file = 'data.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    mileage_data, price_data = load_data(data_file)
    
    min_km, max_km = min(mileage_data), max(mileage_data)
    min_price, max_price = min(price_data), max(price_data)

    normalized_mileage_data = normalize_data(mileage_data, min_km, max_km)
    normalized_price_data = normalize_data(price_data, min_price, max_price)

    learning_rate = 0.1
    iterations = 1000

    print("Training model...")
    theta0, theta1 = train_model(normalized_mileage_data, normalized_price_data, learning_rate, iterations)
    
    save_params(theta0, theta1, min_km, max_km, min_price, max_price)
    
    print("Training complete.")
    print(f"Final theta0: {theta0}")
    print(f"Final theta1: {theta1}")
    print("Thetas and normalization parameters have been saved to 'thetas.txt'.")

if __name__ == "__main__":
    main()