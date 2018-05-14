
#include<iostream>
#include<math.h>

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
using namespace std;

//#define cin infile //一定不能再oj系统中，有错，导致超时等！！！
//C++文件输入
ifstream infile("in.txt", ifstream::in);

#include <limits>
#define INT_MIN     (-2147483647 - 1) /* minimum (signed) int value */
#define INT_MAX       2147483647    /* maximum (signed) int value */

#include <array>
#include <bitset>

// Definition for singly - linked list.
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};


// Definition for a binary tree node.
struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


//1. 对于序列化：使用前序遍历，递归的将二叉树的值转化为字符，并且在每次二叉树的结点
//不为空时，在转化val所得的字符之后添加一个' ， '作为分割。对于空节点则以 '#' 代替。
//2. 对于反序列化：按照前序顺序，递归的使用字符串中的字符创建一个二叉树(特别注意：
//在递归时，递归函数的参数一定要是char ** ，这样才能保证每次递归后指向字符串的指针会
//随着递归的进行而移动
class Solution_Serialize{
public:
	void help_seri(TreeNode* root, string &res)
	{
		if (!root)
		{
			res += '#,';
			return;
		}
		//前序遍历
		//char r[10];
		//sprintf(r, "%d", root->val);
		string r = to_string(root->val);
		res += r;
		res += ',';
		help_seri(root->left, res);
		help_seri(root->right, res);

	}

	char* Serialize(TreeNode *root) {

		if (!root)
		{
			return nullptr;
		}

		string res;

		help_seri(root, res);

		return const_cast<char*>(res.c_str());
	}

	// 由于递归时，会不断的向后读取字符串,所以一定要用**str,以保证得到递归后指针str指向未被读取的字符
	TreeNode* help_deseri(char** str)
	{
		if (**str=='#')
		{
			(*str)++;
			return nullptr;
		}
		int num = 0;
		while (**str != '\0'&&**str != ',')
		{
			num = num * 10 + (**str-'0');
			++(*str);
		}
		TreeNode* node = new TreeNode(num);
		if (**str=='\0')
		{
			return node;
		}
		else
		{
			++(*str);
		}

		node->left = help_deseri(str);
		node->right = help_deseri(str);

		return node;
	}

	TreeNode* Deserialize(char *str) {

		if (str==NULL)
		{
			return nullptr;
		}

		TreeNode* ret = help_deseri(&str);

		return ret;
	}
};


class Solution_37{

public:
	// 方法一：借助stl库函数
	int GetNumberOfK(vector<int> data, int k)
	{
		return count(data.begin(), data.end(), k);
	}

	// 方法二：O(n)遍历

	// 方法三：二分查找
	int GetNumberOfK_2(vector<int> data, int k) {
		int lower = getLower(data, k);
		int upper = getUpper(data, k);
		return upper - lower + 1;
	}

	// 获取k第一次出现的下标
	int getLower(vector<int> data, int k){
		int start = 0, end = data.size() - 1;
		int mid = (start + end) / 2;
		while (start <= end){
			if (data[mid] < k){  
				start = mid + 1;
			}
			else{  // data[mid] >= k
				end = mid - 1;
			}
			mid = (start + end) / 2;
		}
		return start;
	}
	// 获取k最后一次出现的下标
	int getUpper(vector<int> data, int k){
		int start = 0, end = data.size() - 1;
		int mid = (start + end) / 2;
		while (start <= end){
			if (data[mid] <= k){ // data[mid] == k
				start = mid + 1;
			}
			else{
				end = mid - 1;
			}
			mid = (start + end) / 2;
		}
		return end;
	}

	int getFirstK(int* data, int k, int start, int end){
		while (start <= end){
			int mid = start + ((end - start) >> 1);
			if (data[mid] == k){
				if ((mid > 0 && data[mid - 1] != k) || mid == 0)
					return mid;
				else
					end = mid - 1;
			}
			else if (data[mid] > k)
				end = mid - 1;
			else
				start = mid + 1;
		}
		return -1;
	}

	int getLastK(int* data, int length, int k, int start, int end){
		while (start <= end){
			int mid = start + ((end - start) >> 1);
			if (data[mid] == k){
				if ((mid < length - 1 && data[mid + 1] != k) || mid == length - 1)
					return mid;
				else
					start = mid + 1;
			}
			else if (data[mid] < k)
				start = mid + 1;
			else
				end = mid - 1;
		}
		return -1;
	}

	// 标准的二分查找
	int binary_search(int*data, int n, int target)
	{
		int low = 0, high = n - 1;
		while (low<=high)  //必须有等号，避免边界查找不到
		{
			int mid = low + (high - low) / 2;
			if (data[mid]==target)
			{
				return mid;
			}
			else if (data[mid]>target)
			{
				high = mid - 1;
			}
			else
			{
				low = mid + 1;
			}
		}
		return -1;
	}

	int binary_search_test(int*data, int n, int target)
	{
		int low = 0, high = n - 1;
		int mid = low + (high - low) / 2;
		while (low <= high)  //必须有等号，避免边界查找不到
		{		
			if (data[mid] == target)
			{
				return mid;
			}
			else if (data[mid] > target)
			{
				high = mid - 1;
			}
			else
			{
				low = mid + 1;
			}

			mid = low + (high - low) / 2;
		}
		return -1;
	}

	// 用二分法寻找上界
	int binary_search_upperbound(int array[], int low, int high, int target)
	{
		//Array is empty or target is larger than any every element in array 
		if (low > high || target >= array[high]) //控制找不到的情况
			return -1;

		int mid = (low + high) / 2;
		while (high > low)
		{
			if (array[mid] > target)
				high = mid;
			else
				low = mid + 1;

			mid = (low + high) / 2;
		}

		return mid;
	}
	int binary_search_lowerbound(int array[], int low, int high, int target)
	{
		//Array is empty or target is less than any every element in array
		if (high < low || target <= array[low]) 
			return -1;

		int mid = (low + high + 1) / 2; //make mid lean to large side
		while (low < high)
		{
			if (array[mid] < target)
				low = mid;
			else
				high = mid - 1;

			mid = (low + high + 1) / 2;
		}

		return mid;
	}


	int search_rotation(vector<int>& nums, int target) { //旋转数组查找
		int l = 0, r = nums.size() - 1;
		while (l <= r) {
			int mid = (l + r) / 2;
			if (target == nums[mid])
				return mid;
			// there exists rotation; the middle element is in the left part of the array
			if (nums[mid] > nums[r]) {
				if (target < nums[mid] && target >= nums[l])
					r = mid - 1;
				else
					l = mid + 1;
			}
			// there exists rotation; the middle element is in the right part of the array
			else if (nums[mid] < nums[l]) {
				if (target > nums[mid] && target <= nums[r])
					l = mid + 1;
				else
					r = mid - 1;
			}
			// there is no rotation; just like normal binary search
			else {
				if (target < nums[mid])
					r = mid - 1;
				else
					l = mid + 1;
			}
		}
		return -1;
	}
};

int main()
{
	int array[] = { 2, 3, 7, 7, 7, 13, 17 };
	int target = 7;

	Solution_37 su_37;
	int ret1=su_37.GetNumberOfK_2(vector<int>(array,array+7), target); //3

	int ret2 = su_37.binary_search(array,7, target); //3

	int ret3 = su_37.binary_search_test(array, 7, target); //3

	int ret4 = su_37.getUpper(vector<int>(array, array + 7), target); //4

	int ret5 = su_37.binary_search_upperbound(array, 0, 6, target);//5

	int ret6 = su_37.getLower(vector<int>(array, array + 7), target);//2

	int ret7 = su_37.binary_search_lowerbound(array, 0, 6, target);//1

	vector<int> res(array, array + 7);
	int ret8=lower_bound(res.begin(), res.end(), target)-res.begin(); //2

	int ret9 = upper_bound(array, array + 7, target)-array; //5


	Solution_Serialize su_1;
	char str[] = { '3', ',', '9', ',', '20', ',', ' #', '#', '15', ',', '7', ',' };
	TreeNode* ret = su_1.Deserialize(str);
	
	return 0;
}