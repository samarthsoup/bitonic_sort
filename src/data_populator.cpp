#include <iostream>
#include <fstream>
#include <cstdlib>   
#include <ctime>     
#include <sstream>   
#include <string>

int main(int argc, char* argv[]) 
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <number of random numbers> <output file name>\n";
        return 1;
    }

    int n = 0;
    std::istringstream iss(argv[1]);
    if (!(iss >> n) || n <= 0) {
        std::cerr << "invalid number of random numbers: " << argv[1] << "\n";
        return 1;
    }

    std::string filename;

    if (argc >= 3) {
        filename = argv[2];
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".txt") {
            filename += ".txt";
        }
    } else {
        filename = "data/generated_data.txt";
    }


    std::srand(static_cast<unsigned int>(std::time(0)));

    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "error: could not open output file.\n";
        return 1;
    }

    outfile << n << " ";

    for (int i = 0; i < n; ++i) {
        int random_number = std::rand() % 100; 
        outfile << random_number << " ";  
    }

    outfile.close();  
    std::cout << "successfully wrote " << n << " random numbers to " << filename << "\n";

    return 0;
}