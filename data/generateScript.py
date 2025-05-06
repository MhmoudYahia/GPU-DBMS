import pandas as pd
import os

def generate_script(count, name, output_dir):
    # Create a DataFrame with the specified number of rows
    df = pd.DataFrame({
        'id': range(1, count + 1),
        'name': [name] * count,
        'age': [i % 100 for i in range(count)],
        'salary': [i * 1000 for i in range(count)],
    })

    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(output_file, index=False)
    print(f"Generated {count} rows of data for {name} and saved to {output_file}")
if __name__ == "__main__":
    # Example usage
    count = 1000
    name = "users"
    output_dir = "data/input_csvs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    generate_script(count, name, output_dir)