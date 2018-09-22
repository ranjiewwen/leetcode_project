
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


	vector<vector<int>> threeSum1(vector<int> &numbers) {
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

class Solution_28 {
public:
	/**
	* @param matrix: matrix, a list of lists of integers
	* @param target: An integer
	* @return: a boolean, indicate whether matrix contains target
	*/
	bool searchMatrix1(vector<vector<int>> &matrix, int target) {
		// write your code 
		if (matrix.empty())
			return false;
		int row = matrix.size();
		int col = matrix[0].size();

		for (int i = row - 1; i >= 0;) {
			/* code */
			for (int j = 0; j<col;)
			{
				if (matrix[i][j] == target)
					return true;
				else if (matrix[i][j]<target)
					j++;
				else
					i--;
			}
		}
		return false;
	}

	bool binary_search(vector<int>& vec, int target)
	{
		int low = 0;
		int high = vec.size() - 1;
		while (low <= high)
		{
			int mid = low + (high - low) / 2;
			if (vec[mid] == target)
				return true;
			else if (vec[mid]>target)
				high = mid - 1;
			else
				low = mid + 1;
		}
		return false;
	}

	bool searchMatrix(vector<vector<int>> &matrix, int target)
	{
		if (matrix.empty())
			return false;
		int low = 0;
		int high = matrix.size() - 1; //二分找对应行

		int mid = (high + low + 1) / 2;
		while (low < high)
		{
			
			if (matrix[mid][0] == target)
				return true;
			else if (matrix[mid][0]>target)
				high = mid - 1;
			else
				low = mid;

			mid = (high + low + 1) / 2;
		}

		//对应low行二分查找列
		return binary_search(matrix[mid], target);

	}

};
int main()
{
	Solution_59 su_59;
	vector<int> vec_59({ 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 0, 0, -2, 2, -5, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99, 1, 2, 5, 6, 7, 3, 5, 8, -33, -5, -72, 12, -34, 100, 99});
	su_59.threeSumClosest(vec_59,25);


	Solution_28 su_28;
	vector<vector<int>> vec_28 = { {1,4,5}, {6,7,8} };
	su_28.searchMatrix(vec_28, 8);


	return 0;
}