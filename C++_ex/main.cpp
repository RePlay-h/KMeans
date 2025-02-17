

#include "../KMeans_CUDA.hpp"
#include "../CSVParser.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
    
    std::vector<IrisRow> table = parser::ReadIris("../DATA/Iris.csv");

    float *h_params = new float[4 * table.size()];
    int *h_species = new int[table.size()];

    parser::ConvertToArr(table, h_params, h_species);
    
    KMeans model(h_params, 3, table.size(), 4);

    model.fit(25);

    auto preds = model.predict();

    
    delete[] h_params;
    delete[] h_species;

}
