#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <string>

#include "cifar10_reader.h"

class DataLoader {
 private:
  std::string dataset_name;

 public:
  explicit DataLoader(const std::string &dataset_name);
  void 

};

DataLoader::DataLoader(const std::string &dataset_name) {
  this->dataset_name = dataset_name;
}



#endif