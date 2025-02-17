
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

struct IrisRow {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    short int species;
};

namespace parser {

    std::vector<IrisRow> ReadIris(const std::string& filename) {
        std::ifstream f(filename);
        std::string line;

        std::getline(f, line); // get header
        unsigned int num_of_columns = std::count(begin(line), end(line), ',');

        std::vector<IrisRow> table;

        while(std::getline(f, line)) {

            IrisRow row;
            std::string temp; // a cell
            std::stringstream ss(line);

            std::getline(ss, temp, ',');

            std::getline(ss, temp, ',');
            row.sepal_length = stof(temp);

            std::getline(ss, temp, ',');
            row.sepal_width = stof(temp);

            std::getline(ss, temp, ',');
            row.petal_length = stof(temp);

            std::getline(ss, temp, ',');
            row.petal_width = stof(temp);

            std::getline(ss, temp, ' ');

            if(temp == "Iris-setosa") {
                row.species = 0;
            }
            else if(temp == "Iris-versicolor") {
                row.species = 1;
            }
            else {
                row.species = 2;
            }
            

            table.push_back(row); // save a row into table
        }

        return table;
    }    

    void ConvertToArr(std::vector<IrisRow> &table, float *h_params, int *h_species) {

        unsigned int rows = table.size();
        
        for(unsigned int i = 0; i < rows; ++i) {

            h_params[i] = table[i].sepal_length;
            h_params[i + 1] = table[i].sepal_width;
            h_params[i + 2] = table[i].petal_length;
            h_params[i + 3] = table[i].petal_width;

            h_species[i] = table[i].species;
            
        }
    }

}