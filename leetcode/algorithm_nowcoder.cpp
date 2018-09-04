
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

//#define cin infile //һ��������ojϵͳ�У��д����³�ʱ�ȣ�����
//C++�ļ�����
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

class ScaleSort {
public:
	vector<int> sortElement_1(vector<int> A, int n, int k) {
		// write code here
		sort(A.begin(), A.end());
		return A;
	}

	void headAdjust(vector<int> &src,int root,int size) //����ΪС����;���������,����ַ
	{
		int temp = src[root];
		for (int child = 2 * root + 1; child < size;child=2*child+1)
		{
			if (child+1<size&&src[child]>src[child+1]) //�Һ���С
			{
				child++;
			}
			if (temp<src[child])
			{
				break;
			}
			src[root] = src[child];
			root = child;
		}
		src[root] = temp;
		return;
	}

	vector<int> sortElement(vector<int> A, int n, int k) {
		// write code here
		if (n==0||n<k)
		{
			return A;
		}
		vector<int> B; //����k��Ԫ��
		for (int i = 0; i < k;i++)
		{
			B.push_back(A[i]);
		}
		//����С���ѣ��������ѵù���
		for (int i = k / 2-1; i >= 0;i--)
		{
			headAdjust(B,i,k);
		}
		for (int i = k; i < n;i++)
		{
			A[i - k] = B[0]; //�����Ѷ�Ԫ��
			B[0] = A[i];
			headAdjust(B, 0, k);
		}
		//���k��Ԫ�ض�����
		for (int i = n - k; i < n; i++)
		{
			A[i] = B[0];
			swap(B[k-1],B[0]);
			headAdjust(B, 0, --k);
		}
		return A;
	}


};


//int main()
//{
//	ScaleSort s;
//	vector<int> vec = { 2, 1, 4, 3, 6, 5, 8, 7, 10, 9 };
//	vec = s.sortElement(vec,10,2);
//	return 0;
//}