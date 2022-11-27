// Copyright 2014 Nicolas Mellado
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -------------------------------------------------------------------------- //
//
// Authors: Nicolas Mellado
//
// An implementation of the Super 4-points Congruent Sets (Super 4PCS)
// algorithm presented in:
//
// Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing
// Nicolas Mellado, Dror Aiger, Niloy J. Mitra
// Symposium on Geometry Processing 2014.
//
// Data acquisition in large-scale scenes regularly involves accumulating
// information across multiple scans. A common approach is to locally align scan
// pairs using Iterative Closest Point (ICP) algorithm (or its variants), but
// requires static scenes and small motion between scan pairs. This prevents
// accumulating data across multiple scan sessions and/or different acquisition
// modalities (e.g., stereo, depth scans). Alternatively, one can use a global
// registration algorithm allowing scans to be in arbitrary initial poses. The
// state-of-the-art global registration algorithm, 4PCS, however has a quadratic
// time complexity in the number of data points. This vastly limits its
// applicability to acquisition of large environments. We present Super 4PCS for
// global pointcloud registration that is optimal, i.e., runs in linear time (in
// the number of data points) and is also output sensitive in the complexity of
// the alignment problem based on the (unknown) overlap across scan pairs.
// Technically, we map the algorithm as an ‘instance problem’ and solve it
// efficiently using a smart indexing data organization. The algorithm is
// simple, memory-efficient, and fast. We demonstrate that Super 4PCS results in
// significant speedup over alternative approaches and allows unstructured
// efficient acquisition of scenes at scales previously not possible. Complete
// source code and datasets are available for research use at
// http://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/.

#pragma once

#include <list>
#include "gr/accelerators/utils.h"

namespace gr{

/*!
  \brief Multidimensional node used for intersection query

   Each instance store references to the data arrays:
    - const access to the input points
    - writing access to the dereferencing id array

   The working dimension is deduced from the Point class
 */
template <class Point,
          int _dim,
          typename Scalar,
          class _PointContainer = std::vector<Point>,
          class _IdContainer    = std::vector<unsigned int> >
class NdNode{
public:
  enum { Dim = _dim };
  typedef _PointContainer PointContainer;
  typedef _IdContainer    IdContainer;

protected:
  const PointContainer& _points; //!< Input points. Needed to compute split
  IdContainer&          _ids;    //!< Ids used to access _points[_ids[i]]
  Point _center;                 //!< Center of the node in the nd-space
  unsigned int _begin,           //!< First element id in the node
               _end;             //!< Last element id in the node

public:
  inline NdNode(const PointContainer& points,
                IdContainer& ids,
                const Point& p     = Point::Zero(),
                unsigned int begin = 0,
                unsigned int end   = 0)
    : _points(points), _ids(ids), _center(p), _begin(begin), _end(end) {}

  inline NdNode(const NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer>&other)
    : _points(other._points), _ids(other._ids),
      _center(other._center),
      _begin (other._begin),  _end(other._end) {}

  inline NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer>& operator=
  (const NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer>& rhs)
  {
    //_points = rhs._points; // Nodes must be working on the same container
    _ids    = rhs._ids;
    _center = rhs._center;
    _begin  = rhs._begin;
    _end    = rhs._end;

    return *this;
  }

  inline ~NdNode(){}

  //! \brief Length of the range covered by this node in the id array
  inline int rangeLength()         const { return int(_end) - int(_begin); }
  //! \brief First position in the id array defining the current instance range
  inline unsigned int rangeBegin() const { return _begin; }
  //! \brief Last position in the id array defining the current instance range
  inline unsigned int rangeEnd()   const { return _end; }
  //! \brief Access to the i-eme point in the node. Range limits are NOT tested!
  inline const Point& pointInRange(unsigned int i) const
  { return _points[_ids[i+_begin]]; }
  //! \brief Access to the i-eme point in the node. Range limits are NOT tested!
  inline unsigned int idInRange(unsigned int i) const
  { return _ids[i+_begin]; }

  //! \brief Center of the current node
  inline const Point& center()     const { return _center; }

  //! Split the node and compute child nodes \note Childs are not stored
  //! \todo See how to introduce dimension specialization (useful for 2D and 3D)
  void
  split(
    std::vector< NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer> > &childs,
    Scalar rootEdgeHalfLength );

  inline static
  NdNode <Point, _dim, Scalar, _PointContainer, _IdContainer>
  buildUnitRootNode(const PointContainer& points,
                    IdContainer& ids)
  {
    return NdNode <Point, _dim, Scalar, _PointContainer, _IdContainer> (
      points, ids, Point::Ones() / 2., 0, ids.size());
  }

  private:
  inline unsigned int _split(int start, int end,
                             unsigned int dim,
                             Scalar splitValue);
};


template <class Point,
          int _dim,
          typename Scalar,
          class _PointContainer,
          class _IdContainer>
unsigned int
NdNode< Point, _dim, Scalar, _PointContainer, _IdContainer>::_split(
  int start,
  int end,
  unsigned int dim,
  Scalar splitValue)
{
  int l(start), r(end-1);
  // 在 xyz 轴上，分别进行 8 叉树划分点
  // 将小于 splitvalue 的点放到左边，大于的放到右边
  // 在第一次划分时，中心点就是 0.5，会处理所有点
  for ( ; l<r ; ++l, --r)
  {
    while (l < end && _points[_ids[l]][dim] < splitValue)
        l++;
    while (r >= start && _points[_ids[r]][dim] >= splitValue)
        r--;
    if (l > r)
        break;
    std::swap(_ids[l],_ids[r]);
  }
  if(l>=end) return end; // 如果所有点都在中值左边，说明所有点都被分到一侧了
  return _points[_ids[l]][dim] < splitValue ? l+1 : l;
}

/*!
   \brief Split the current node in 2^Dim childs using a regular grid.
   The IdContainer is updated to partially sort the input ids wrt to the childs
   range limits.

   \param ChildContainer in:
 */
template <class Point,
          int _dim,
          typename Scalar,
          class _PointContainer,
          class _IdContainer>
void
NdNode< Point, _dim, Scalar, _PointContainer, _IdContainer>::split(
    std::vector< NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer> > &childs,
    Scalar rootEdgeHalfLength )
{
  // 这个算法主要目的是分割，将当前的 box 进行八叉划分
  // 首先将 xyz 投影到同一维度上，用 中值 的左右两侧表示第几块
  // 首先按 x 分成 2 段
  // 然后按 y 再将 x 的2段分为 4 段
  // 同样 z 会得到 8 段
  typedef NdNode<Point, _dim, Scalar, _PointContainer, _IdContainer> Node;

  //! Compute number of childs at compile time
  const int nbNode = Utils::POW(int(2),int(Dim));
  const int offset = childs.size();

  // init all potential nodes using the root values
  childs.resize(offset+nbNode, *this);

  /// Split successively along all the dimensions of the ambiant space
  /// This algorithm cannot be parallelized
  // Dim: 这个分点算法可以适用于多维 >3 情况
  // d: xyz 维度
  for(unsigned int d = 0; d < Dim; d++){
    // 2, 4, 8
    const unsigned int nbInterval   = Utils::POW(int(2),int(d+1)); // can be deduced at
    // 1, 2, 4
    const unsigned int nbSplit      = nbInterval/2;         // compile time with
    // 8, 4, 2
    const unsigned int intervalNode = nbNode / nbSplit;     // loop unrollement
    // 4, 2, 1
    const unsigned int midNode      = nbNode / nbInterval;

    /// Iterate over all splits and compute them for the current dimension
    for(unsigned int s = 0; s != nbSplit; s++)    {
      const unsigned int beginNodeId =  s    * intervalNode + offset;
      const unsigned int endNodeId   = (s+1) * intervalNode + offset;

      Scalar currentCenterD = childs[beginNodeId]._center[d];

      const unsigned int splitId = _split(childs[beginNodeId]._begin,
                                          childs[endNodeId-1]._end,
                                          d,
                                          currentCenterD);
      // 普通的八叉树中心计算
      const Scalar beforeCenterD = currentCenterD - rootEdgeHalfLength/Scalar(2);
      const Scalar afterCenterD  = currentCenterD + rootEdgeHalfLength/Scalar(2);

      /// Transmit the split to the related nodes
      // 将当前计算的部分点云分为两段，给下一个维度使用
      for (unsigned int i = beginNodeId; i != beginNodeId + midNode; i++){
        childs[i]._center[d] = beforeCenterD;
        childs[i]._end       = splitId;
      }

      for (unsigned int i = beginNodeId + midNode; i != endNodeId; i++){
        childs[i]._center[d] = afterCenterD;
        childs[i]._begin     = splitId;
      }
    }
  }

  // Remove childs not containing any element
  if (!childs.empty()) {
      childs.erase(std::remove_if(childs.begin(), childs.end(), [](const Node& c)
      { return c.rangeLength() == 0; }), childs.end());
  }
}

} // namespace gr


