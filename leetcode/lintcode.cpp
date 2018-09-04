
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


int main()
{
	Solution_59 su_59;
	vector<int> vec_59({ 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 0, 0, -2, 2, -5, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99});
	su_59.threeSumClosest(vec_59,25);

	return 0;
}