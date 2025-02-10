

#include "../KMeans_CUDA.hpp"
#include "../CSVParser.hpp"

int main(int argc, char *argv[]) {
    
    std::vector<IrisRow> table = parser::ReadIris("../DATA/Iris.csv");

    for(IrisRow r : table) {
        auto&[a,b,c,d,e] = r;

        std::cout << a << ' ' << b << ' ' << c << ' ' << d << ' '<< e << '\n';
    }

    float *h_params = new float[4 * table.size()];
    short int *h_species = new short int[table.size()];

    parser::ConvertToArr(table, h_params, h_species);
    
    delete[] h_params;
    delete[] h_species;

}
