#include "Atlas.h"

namespace ORB_SLAM2
{

Atlas::Atlas() : mpCurrentMap(nullptr)
{
}

Atlas::Atlas(Map* pInitialMap) : mpCurrentMap(pInitialMap)
{
    if (pInitialMap) {
        mspMaps.insert(pInitialMap);
    }
}

void Atlas::AddMap(Map* pMap)
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mspMaps.insert(pMap);
}

Map* Atlas::GetCurrentMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap;
}

void Atlas::SetCurrentMap(Map* pMap)
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mpCurrentMap = pMap;
    mspMaps.insert(pMap);
}

std::set<Map*> Atlas::GetAllMaps()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mspMaps;
}

int Atlas::CountMaps()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mspMaps.size();
}

void Atlas::clear()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    for (auto it = mspMaps.begin(); it != mspMaps.end(); ++it)
    {
        (*it)->clear();
        delete *it;
    }
    mspMaps.clear();
    mpCurrentMap = nullptr;
}

} //namespace ORB_SLAM2
