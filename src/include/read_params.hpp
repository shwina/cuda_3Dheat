# include <fstream>
# include <iostream>
# include <yaml-cpp/yaml.h>

class Grid
{
public:
    int N_x, N_y, N_z;
    float L_x, L_y, L_z;
};

class Sim
{
public:
    int nsteps;
    float alpha, dt;
};

namespace YAML
{
    template<>
    struct convert<Grid>{
        static Node encode(const Grid& g){
            Node node;
            node["N_x"] = g.N_x;
            node["N_y"] = g.N_y;
            node["N_z"] = g.N_z;
            node["L_x"] = g.L_x;
            node["L_y"] = g.L_y;
            node["L_z"] = g.L_z;
            return node;
        }
    
        static bool decode(const Node& node, Grid& g){
            g.N_x = node["N_x"].as<int>();
            g.N_y = node["N_y"].as<int>();
            g.N_z = node["N_z"].as<int>();
            g.L_x = node["N_x"].as<float>();
            g.L_y = node["N_y"].as<float>();
            g.L_z = node["N_z"].as<float>();
            return true;
        }
    };

    template<>
    struct convert<Sim>{
        static Node encode(const Sim& s){
            Node node;
            node["alpha"] = s.alpha;
            node["dt"]    = s.dt;
            node["nsteps"]= s.nsteps;
            return node;
        }
    
        static bool decode(const Node& node, Sim& s){
            s.alpha  = node["alpha"].as<float>();
            s.dt     = node["dt"].as<float>();
            s.nsteps = node["nsteps"].as<int>();
            return true;
        }
    };
}
