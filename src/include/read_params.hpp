// This is a messy implementation. Think about cleaning 
// up.


# include <fstream>
# include <iostream>
# include <yaml-cpp/yaml.h>

struct Grid_params{
    int N_x, N_y, N_z;
    float L_x, L_y, L_z;
};

struct Sim_params{
    int nsteps;
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

void operator >> (const YAML::Node& node, Sim_params& sim_params){
    node["alpha"] >> sim_params.alpha;
    node["dt"] >> sim_params.dt;
    node["nsteps"] >> sim_params.nsteps;
}


std::ifstream fin("params.yml");
YAML::Parser parser(fin);
YAML::Node doc;
parser.GetNextDocument(doc);

Grid_params grid;
Sim_params  sim;

doc["grid_params"] >> grid;
doc["sim_params"] >> sim;

int N_x, N_y, N_z, nsteps;
float L_x, L_y, L_z, alpha, dt;
float dx, dy, dz;

N_x = grid.N_x;
N_y = grid.N_y;
N_z = grid.N_z;
L_x = grid.L_x;
L_y = grid.L_y;
L_z = grid.L_z;
alpha   = sim.alpha;
dt      = sim.dt;
nsteps  = sim.nsteps;

dx = L_x/N_x;
dy = L_y/N_y;
dz = L_z/N_z;