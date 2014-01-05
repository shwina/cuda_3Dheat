# include <iostream>
# include <stdio.h>
# include <yaml-cpp/yaml.h>
# include <fstream>


struct Grid_params{
    int N_x, N_y, N_z;
    float L_x, L_y, L_z;
};

struct Sim_params{
    float alpha, dt;
};


void operator >> (const YAML::Node& node, Grid_params& grid_params){
    node["N_x"] >> grid_params.N_x;
    node["N_y"] >> grid_params.N_y;    
    node["N_z"] >> grid_params.N_z; 
    node["L_x"] >> grid_params.L_x;
    node["L_y"] >> grid_params.L_y;
    node["L_z"] >> grid_params.L_z;
}

void operator >> (const YAML::Node& node, Sim_params& grid_params){
    node["alpha"] >> grid_params.alpha;
    node["dt"] >> grid_params.dt;    
}


int main(){

    std::ifstream fin("params.yml");
    YAML::Parser parser(fin);
    YAML::Node doc;
    parser.GetNextDocument(doc);
    
    Grid_params grid;
    Sim_params  sim;

    doc["grid_params"] >> grid;
    doc["sim_params"] >> sim;
     
    std::cout << grid.N_x << std::endl;
    std::cout << sim.alpha << std::endl;
    return 0;

}
