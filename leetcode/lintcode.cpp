
#include <vector>
#include <string>
#include <deque>
#include <stack>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

#include <algorithm>
#include <functional>
#include <numeric> //accmulate

#include <iterator> //ostream_iterator
#include <fstream>
#include <iomanip>  //setprecision() setw()

#include <memory> 
#include <cstring> //memset
#include <sstream>

#include <limits>
#define INT_MIN     (-2147483647 - 1) /* minimum (signed) int value */
#define INT_MAX       2147483647    /* maximum (signed) int value */
#define eps 1e-6
using namespace std;

#define cin infile //一定不能再oj系统中，有错，导致超时等！！！
//C++文件输入
ifstream infile("in.txt", ifstream::in);
//freopen("in", "r", stdin);
typedef long long ll;


#include<iostream>
#include<vector>
using namespace std;


class Solution_59 {
public:
	/**
	* @param numbers: Give an array numbers of n integer
	* @param target: An integer
	* @return: return the sum of the three integers, the sum closest target.
	*/
	int threeSumClosest(vector<int> &numbers, int target) {
		// write your code here
		if (numbers.size()<3)
			return target;
		int a, b, c;
		sort(numbers.begin(), numbers.end());

		int ret = INT_MIN;
		int diff = INT_MAX;
		for (int i = 0; i<numbers.size() - 2; i++)
		{
			a = i;
			b = i + 1;
			c = numbers.size() - 1;

			while (b<c)
			{
				int sum = numbers[a] + numbers[b] + numbers[c];
				if (sum>=target)
					c--;
				else if (sum<target) //居然sum==target有错误！
					b++;
				/*else
					return 0;*/
				if (abs(sum - target)<diff)
				{
					ret = sum;
					diff = abs(sum - target);
				}
			}
		}
		return ret;
	}
};

class Solution_57 {
public:
	/**
	* @param numbers: Give an array numbers of n integer
	* @return: Find all unique triplets in the array which gives the sum of zero.
	*/

	vector<vector<int>> threeSum(vector<int> &numbers) {
		// write your code here
		vector<vector<int>> ret;
		if (numbers.size() < 3)
			return ret;
		sort(numbers.begin(), numbers.end());
		vector<int> vec;
		for (int i = 0; i < numbers.size() - 2; i++)
		{
			if (i >= 1 && numbers[i - 1] == numbers[i]) //去重复元素
			{
				continue;
			}

			int b = i + 1;
			int c = numbers.size() - 1;

			while (b<c)
			{
				while (b<c&&c<numbers.size() - 1 && numbers[c + 1] == numbers[c]) //去重只需要在查找到过后进行，所以放在下面else里面，容易理解一些
					c--;
				while (b<c&&b - 1>i&&numbers[b - 1] == numbers[b])
					b++;
				if (b == c)
					continue;

				int sum = numbers[i] + numbers[b] + numbers[c];
				if (sum>0)
				{
					c--;
				}
				else if (sum < 0)
				{
					b++;
				}
				else
				{
					vec.clear();
					vec.push_back(numbers[i]);
					vec.push_back(numbers[b]);
					vec.push_back(numbers[c]);
					ret.push_back(vec);
					//break; //bug
					c--;
					b++;
				}
			}
		}
		return ret;
	}


	vector<vector<int>> threeSum(vector<int> &numbers) {
		// write your code here
		vector<vector<int>> ret;
		if (numbers.size() < 3)
			return ret;
		sort(numbers.begin(), numbers.end());
		vector<int> vec;
		for (int i = 0; i < numbers.size() - 2; i++)
		{
			if (i >= 1 && numbers[i - 1] == numbers[i]) //去重复元素
			{
				continue;
			}

			int b = i + 1;
			int c = numbers.size() - 1;

			while (b<c)
			{

				int sum = numbers[i] + numbers[b] + numbers[c];
				if (sum>0)
				{
					c--;
				}
				else if (sum < 0)
				{
					b++;
				}
				else
				{
					vec.clear();
					vec.push_back(numbers[i]);
					vec.push_back(numbers[b]);
					vec.push_back(numbers[c]);
					ret.push_back(vec);
					//break; //bug
					c--;
					b++;

					while (b < c&&numbers[c + 1] == numbers[c])
						c--;
					while (b < c&&numbers[b - 1] == numbers[b])
						b++;

				}
			}
		}
		return ret;
	}
};

int main()
{
	Solution_59 su_59;
	vector<int> vec_59({ 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 0, 0, -2, 2, -5, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99});
	su_59.threeSumClosest(vec_59,25);

	return 0;
}