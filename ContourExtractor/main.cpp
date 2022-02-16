#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<algorithm>
#include<vector>
#include<iterator>
#include<random>

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#include"emst.hpp"

using namespace std;

cv::Mat mat;
cv::Mat dst;
cv::Mat edgesImage;

int thresh = 100;

void OnChange(int, void*)
{
    cv::threshold(mat, dst, thresh, 255, cv::THRESH_BINARY);
    cv::imshow("output", dst);
}

int thresh1 = 100;
int thresh2 = 300;

void OnThreshChange(int, void*)
{
    cv::Canny(mat, edgesImage, thresh1, thresh2);
    cv::imshow("output", edgesImage);
}

vector<cv::Point> samples;

vector<vector<size_t>> g;

int l, ld, pair_dis;
Edge pair0;

void dfs(int i, int parent)
{
    int farthest_leaf = i;
    int farthest_leaf_dis = 0;
    Edge farthest_leaf_pair;
    int farthest_leaf_pair_dis = -1;
    vector<Edge> leave_dis;
    for (size_t index = 0; index < g[i].size(); index++)
    {
        size_t j = g[i][index];
        if (j == parent)
            continue;
        dfs(j, i);
        leave_dis.push_back(Edge(ld + 1, l));
        if (ld + 1 > farthest_leaf_dis)
        {
            farthest_leaf_dis = ld + 1;
            farthest_leaf = l;
        }
        if (farthest_leaf_pair_dis < pair_dis)
        {
            farthest_leaf_pair = pair0;
            farthest_leaf_pair_dis = pair_dis;
        }
    }

    if (leave_dis.size() >= 2)
    {
        sort(leave_dis.begin(), leave_dis.end(), [](const Edge& a, const Edge& b)
            {
                return a < b;
            });
        Edge e1 = leave_dis[leave_dis.size() - 2]; 
        Edge e2 = leave_dis[leave_dis.size() - 1];
        
        if ((int)e1.first + (int)e2.first > farthest_leaf_pair_dis)
        {
            farthest_leaf_pair_dis = e1.first + e2.first;
            farthest_leaf_pair = Edge(e1.second, e2.second);
        }
    }

    l = farthest_leaf;
    ld = farthest_leaf_dis;
    pair0 = farthest_leaf_pair;
    pair_dis = farthest_leaf_pair_dis;

    return;
}

void find_farthest_leaf_pair(size_t& st, size_t& ed)
{
    for (size_t i = 0; i < g.size(); i++)
    {
        if (g[i].size() != 0)
        {
            dfs(i, -1);
            if (g[i].size() == 1 && ld > pair_dis)
            {
                st = i;
                ed = l;
                return;
            }
            st = pair0.first;
            ed = pair0.second;
            return;
        }
    }
}

vector<size_t> vis;
size_t st;
size_t ed;

bool in(size_t number)
{
    for (size_t i = 0; i < vis.size(); i++)
    {
        if (vis[i] == number)
            return true;
    }
    return false;
}

bool dfs2(size_t i)
{
    vis.push_back(i);
    if (i == ed)
    {
        return true;
    }
    for (size_t j = 0; j < g[i].size(); j++)
    {
        if (!in(g[i][j]))
        {
            if (dfs2(g[i][j]))
            {
                size_t temp = g[i][j];
                g[i][j] = g[i][g[i].size() - 1];
                g[i][g[i].size() - 1] = temp;
                return true;
            }
        }
    }
    return false;
}

vector<cv::Point2d> res;

bool dfs3(size_t i)
{
    vis.push_back(i);
    res.push_back(samples[i]);
    if (i == ed)
    {
        return true;
    }
    bool leaf = true;
    for (size_t index = 0; index < g[i].size(); index++)
    {
        size_t j = g[i][index];
        if (!in(j))
        {
            leaf = false;
            if (dfs3(j))
            {
                return true;
            }
        }
    }
    if (!leaf)
    {
        res.push_back(samples[i]);
    }
    return false;
}

inline double dist(const cv::Point2d& p, const cv::Point2d& first, const cv::Point2d& second)
{
    double dy = second.y - first.y;
    double dx = second.x - first.x;
    double bdx = dx * first.y - dy * first.x;
    double d1 = abs(dy * p.x - dx * p.y + bdx);
    double d2 = sqrt(dy * dy + dx * dx);
    return d1 / d2;
}

void RDP(const vector<cv::Point2d>& points, double epsilon, vector<cv::Point2d>& out)
{
    double dmax = 0.0;
    size_t index = 0;
    size_t end = points.size() - 1ULL;

    for (size_t i = 1ULL; i < end; i++)
    {
        double d = dist(points[i], points[0], points[end]);
        if (d > dmax)
        {
            index = i;
            dmax = d;
        }
    }

    if (dmax > epsilon)
    {
        vector<cv::Point2d> recResults1;
        vector<cv::Point2d> recResults2;
        vector<cv::Point2d> firstLine(points.begin(), points.begin() + index + 1ULL);
        vector<cv::Point2d> lastLine(points.begin() + index, points.end());
        RDP(firstLine, epsilon, recResults1);
        RDP(lastLine, epsilon, recResults2);

        out.assign(recResults1.begin(), recResults1.end() - 1ULL);
        out.insert(out.end(), recResults2.begin(), recResults2.end());

    }
    else
    {
        out.clear();
        out.push_back(points[0]);
        out.push_back(points[end]);
    }
}


int main()
{
    string fileName;
    cout << "文件名称:";
    getline(cin, fileName);

    mat = cv::imread(fileName);

    cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
    
    GaussianBlur(mat, mat, cv::Size(3, 3), 0);

    cv::Canny(mat, edgesImage, thresh1, thresh2);

    cv::imshow("output", edgesImage);

    cv::createTrackbar("thresh1", "output", &thresh1, 1000, OnThreshChange);
    cv::createTrackbar("thresh2", "output", &thresh2, 1000, OnThreshChange);

    OnThreshChange(0, 0);

    cv::waitKey(0);

    vector<vector<cv::Point>> contours;

    cv::findContours(edgesImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);//这里选择保存所有点

    vector<cv::Point> contoursPoints;

    for (size_t i = 0; i < contours.size(); i++)
    {
        for (size_t j = 0; j < contours[i].size(); j++)
        {
            contoursPoints.push_back(contours[i][j]);
        }
    }
    
    cout << "样本点数:" << contoursPoints.size() << "\n";

    sample(contoursPoints.begin(), contoursPoints.end(), back_inserter(samples), contoursPoints.size() *2ULL/3ULL, mt19937{ random_device{}() });//随机取样50%的点

    cout << "随机取样:" << samples.size() << "\n";

   vector<Point<2>> points(samples.size());

   for (size_t i = 0; i < points.size(); i++)
   {
       points[i][0] = samples[i].x;
       points[i][1] = samples[i].y;
   }

   KdTreeSolver<2> solver(points);

   vector<Edge> edges = solver.get_solution();

   cv::Mat spanningTreeImage;
   spanningTreeImage.create(mat.size(), mat.type());

   g = vector<vector<size_t>>(samples.size());

   for (size_t i = 0; i < edges.size(); i++)
   {
       g[edges[i].first].push_back(edges[i].second);
       g[edges[i].second].push_back(edges[i].first);

       cv::line(spanningTreeImage, samples[edges[i].first], samples[edges[i].second], cv::Scalar(255, 255, 255));
   }

   cv::imshow("Minimum Spanning Tree", spanningTreeImage);

   cv::waitKey(0);

   find_farthest_leaf_pair(st, ed);

   dfs2(st);
   
   vis.clear();

   dfs3(st);

   vector<cv::Point2d> dft_results;

   for (size_t i = 0; i < res.size(); i++)
   {
       res[i].y = mat.rows - res[i].y;
   }

   cout << "轨迹点数:" << res.size() << "\n";

   vector<cv::Point2d> rdpResults;

   RDP(res, 0.5, rdpResults);

   cout << "拉默-道格拉斯-普克算法后减少至:" << rdpResults.size() << "\n";

   cv::dft(rdpResults, dft_results, cv::DFT_SCALE);

   ofstream file = ofstream("dft_data.json", ios::trunc | ios::out);

   file << "[";

   for (size_t i = 0; i < dft_results.size() - 1; i++)
   {
       file << "{\"x\":" << dft_results[i].x << ",\"y\":" << dft_results[i].y << "},";
   }

   file << "{\"x\":" << dft_results[dft_results.size() - 1].x << ",\"y\":" << dft_results[dft_results.size() - 1].y << "}";

   file << "]";

   file.close();

   cv::waitKey(0);

	return 0;
}