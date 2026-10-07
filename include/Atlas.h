#ifndef ATLAS_H
#define ATLAS_H

#include "Map.h"
#include <set>
#include <mutex>
#include <vector>

namespace ORB_SLAM2
{

class Atlas
{
public:
    Atlas();
    Atlas(Map* pInitialMap);
    
    void AddMap(Map* pMap);
    Map* GetCurrentMap();
    
    void SetCurrentMap(Map* pMap);
    
    std::set<Map*> GetAllMaps();
    int CountMaps();
    
    void clear();

protected:
    std::set<Map*> mspMaps;
    Map* mpCurrentMap;
    std::mutex mMutexAtlas;
};

} //namespace ORB_SLAM2

#endif // ATLAS_H
