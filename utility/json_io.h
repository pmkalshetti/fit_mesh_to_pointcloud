#include <json/json.h>
#include <Eigen/Eigen>

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

Eigen::VectorXd read_vec_from_json(const std::filesystem::path &path_json, const std::string vec_name)
{
    std::ifstream file_json(path_json, std::ifstream::binary);
    if (!file_json.is_open())
    {
        std::cerr << "Unable to open file " << path_json << "\n";
    }
    
    Json::Value root_object;
    file_json >> root_object;

    const Json::Value vec_value = root_object[vec_name];
    Eigen::VectorXd vec(vec_value.size());
    for (int i{0}; i < vec.size(); ++i)
        vec(i) = vec_value[i].asDouble();    
    
    file_json.close();

    return vec;
}