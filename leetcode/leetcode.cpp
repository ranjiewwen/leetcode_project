
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

using namespace std;

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


//Definition for binary tree with next pointer.
struct TreeLinkNode {
	int val;
	TreeLinkNode *left, *right, *next;
	TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};

// Definition for singly - linked list with a random pointer.
struct RandomListNode {
	int label;
	RandomListNode *next, *random;
	RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};


//Definition for undirected graph.
struct UndirectedGraphNode {
	int label;
	vector<UndirectedGraphNode *> neighbors;
	UndirectedGraphNode(int x) : label(x) {};
};


//evluate reverse polish notation accept
class Solution_150 {
public:
	int str2digit(string str)
	{


		int ret = 0;

		const char *src = str.c_str();
		ret = atoi(src);

		//for (int i = 0; i < str.size();i++)
		//{
		//	if ()
		//	{
		//	}
		//}
	}

	string digit2str(int data)
	{
		string ret;
		char temp[128] = " ";
		sprintf_s(temp, "%d", data); //./solution.h:25:3: error: use of undeclared identifier 'sprintf_s'

		ret = temp;
		return ret;
	}
	int evalRPN(vector<string> &tokens) {
		stack<string> sta;
		int oper1, oper2,oper3;
		for (unsigned int i = 0; i < tokens.size();i++)
		{
			if (tokens[i] == "+")
			{
				oper2 = str2digit(sta.top());
				sta.pop();
				oper1 = str2digit(sta.top());
				sta.pop();

				oper3 = oper1 + oper2;
				sta.push(digit2str(oper3));

			}else if (tokens[i]=="-")
			{
				oper2 = str2digit(sta.top());
				sta.pop();
				oper1 = str2digit(sta.top());
				sta.pop();

				oper3 = oper1 - oper2;
				sta.push(digit2str(oper3));
			}
			else if (tokens[i]=="*")
			{
				oper2 = str2digit(sta.top());
				sta.pop();
				oper1 = str2digit(sta.top());
				sta.pop();

				oper3 = oper1 * oper2;
				sta.push(digit2str(oper3));
			}
			else if (tokens[i]=="/")
			{
				oper2 = str2digit(sta.top());
				sta.pop();
				oper1 = str2digit(sta.top());
				sta.pop();

				oper3 = oper1 / oper2;  //除零保护
				sta.push(digit2str(oper3));
			}
			else
			{
				sta.push(tokens[i]);
			}
		}
		return str2digit(sta.top());
	}
};

// sort list
class Solution_148 {
public:

	ListNode* findMiddle(ListNode* head)
	{
		ListNode* slow = head;
		ListNode* fast = head->next;
		while (fast&&fast->next) //bug1 ||
		{
			slow = slow->next;
			fast = fast->next->next;
		}
		return slow;
	}

	ListNode* MergeList(ListNode* left, ListNode*right)
	{
		if (left==NULL)
		{
			return right;
		}
		if (right==NULL)
		{
			return left;
		}

		ListNode* temp = new ListNode(0);
		ListNode* temp_head = temp;  //bug2 头指针移动
		while (left&&right)
		{
			if (left->val<right->val)
			{
				temp->next = left;  //bug3 顺序反了
				left = left->next;
				
			}
			else
			{
				temp->next = right;
				right = right->next;
			}
			temp = temp->next;
		}

		if (left) //bug4 if->while
		{
			temp->next = left;
		}
		if (right)
		{
			temp->next = right;
		}

		return temp_head->next;

	}
	ListNode* sortList(ListNode* head) {

		if (!head||!head->next)  // Line 57: member access within null pointer of type 'struct ListNode'
		{
			return head;  //bug5 忘记取！
		}
		
		ListNode *middle = findMiddle(head);


		ListNode* left=sortList(middle->next);

		middle->next = NULL;
		ListNode* right = sortList(head);

		ListNode *ret=MergeList(left, right);
		return  ret;
	}
};

//postorder traversal
//* 核心思想是用栈做辅助空间，先从根往左一直入栈，直到为空，然后判断栈顶元素的右孩子，如果不为空且未被访问过，
//* 则从它开始重复左孩子入栈的过程；否则说明此时栈顶为要访问的节点（因为左右孩子都是要么为空要么已访问过了），
//* 出栈然后访问即可，接下来再判断栈顶元素的右孩子...直到栈空。
class Solution_145 {
public:
	vector<int> postorderTraversal(TreeNode *root) {
		stack<TreeNode*> sta;

		TreeNode* cur;
		TreeNode* pre=NULL;
		vector<int> vec;
		if (root==NULL)
		{
			return vec;
		}
		if (root->left == NULL&&root->right == NULL)
		{
			vec.push_back(root->val);
			return vec;
		}
		else
		{
			sta.push(root); //保证左节点先于右节点被访问，根节点后于左右节点被访问
			while (!sta.empty())
			{
			    cur = sta.top();			

				if ((cur->left==NULL&&cur->right==NULL)||pre!=NULL&&(pre==cur->left||pre==cur->right))
				{ //访问该节点
					vec.push_back(cur->val);
					sta.pop();
					pre = cur;
				}
				else
				{
					if (cur->right) //先压右孩子，再压左孩子
					{
						sta.push(cur->right);
					}
					if (cur->left)
					{
						sta.push(cur->left);
					}
				}
			}
		}
		return vec;
	}
};

//preorder tranversal
class Solution_144 {
public:
	vector<int> preorderTraversal(TreeNode *root) {
		vector<int> vec;
		stack<TreeNode*> sta;
		if (root)
		{
			/*vec.push_back(root->val);
			sta.push(root);
			root = root->left;*/
			while (root||!sta.empty()) // && bug1  //树不为空或者栈不为空，继续循环
			{
				while (root)
				{
					vec.push_back(root->val); //第一次遍历根节点
					sta.push(root);
					root = root->left;
				}
				if (!sta.empty())
				{
					TreeNode* temp = sta.top();
					sta.pop();
					root = temp->right;
				}
			}
		}
		return vec;
	}
};

// insertion sort list
class Solution_147 {
public:
	ListNode *insertionSortList(ListNode *head) {
	//思路:新建一个链表, 遍历原链表，将每个节点加入新链表正确的位置

		ListNode* dummy = new ListNode(0);
		ListNode* cur = head; //原链表当前位置
		ListNode* next = NULL;  //记录原链表下一节点
		ListNode* pre = dummy; //新链表需要比较的前后节点

		while (cur)
		{
			next = cur->next;
			while (pre->next&& pre->next->val < cur->val)
			{
				pre = pre->next;
			}

			cur->next = pre->next; //将当前节点独立出来
			pre->next = cur;
			pre = dummy; //回到起点

			cur = next;    //处理下一个节点
		}
		return dummy->next;
	}
};

//reorder list
class Solution_143 {
public:

	ListNode* findMiddle(ListNode* head)
	{
		ListNode* slow=head;
		ListNode* fast = head;
		while ( fast && fast->next)
		{
			slow = slow->next;
			fast = fast->next->next;
		}
		return slow;
	}

	ListNode* reverse_list(ListNode*head)
	{
		ListNode* pre = head;
		ListNode* temp = head->next;
		ListNode* cur = temp;

		pre->next = NULL;
		
		while (cur)
		{
			temp = cur->next;
			cur->next = pre;

			pre = cur;
			cur = temp;
		}
		return pre;
	}

	void reorderList(ListNode *head) {  //void

		if (head==NULL||head->next==NULL||head->next->next==NULL)
		{
			return;
		}
		//快慢指针找到中间节点，将后面的链表反转（前插法），合并链表
		//另：题目要求是就地解决，应该是不能用辅助栈之类的
		ListNode* middel = findMiddle(head);

		//反转链表
		ListNode* last = reverse_list(middel->next);
		middel->next = NULL;

		ListNode* temp = NULL;
		ListNode* cur = head;
		while (last)  //防止形成环 middel->next = NULL;
		{
			temp = last->next;
			last->next = cur->next;

			cur->next = last;

			last = temp;
			cur = cur->next->next;
		}

		return;
	}
};

// Linked List cycle-ii
class Solution_142 {
public:
	ListNode *detectCycle(ListNode *head) {
		if (!head||!head->next)
		{
			return NULL;
		}
		ListNode* slow=head;
		ListNode* fast=head;

		bool isCycle = false;
		while (fast&&fast->next)
		{
			slow = slow->next;
			fast = fast->next->next;  // palce front
			if (slow==fast)
			{
				isCycle = true;
				break; //has cycle 第一次退出
			}
			
		}

		if (!isCycle)
		{
			return NULL;
		}
		else
		{
			ListNode* first = head;
			while (first!=slow)
			{
				first = first->next;
				slow = slow->next;
			}
		}
		return slow;
	}
};

// Linked List Cycle
class Solution_141 {
public:
	bool hasCycle(ListNode *head) {
		if (!head||!head->next)
		{
			return false;
		}
		ListNode* slow = head;
		ListNode* fast = head->next;

		while (fast && fast->next) //不能用slow
		{
			if (slow==fast)
			{
				return true;
			}
			else
			{
				slow = slow->next;
				fast = fast->next->next;
			}
		}
		return false;
	}
};

// world break -i
class Solution_139 {
public:
	bool wordBreak(string s, unordered_set<string> &dict) {
		if (dict.size()==0)
		{
			return false;
		}
		//动态规划
		vector<bool> dp(s.size() + 1, false);
		dp[0] = true;

		for (int i = 1; i <= s.size();i++) //第一层遍历，s的每个位置是否可分成字典元素
		{
			for (int j = i - 1; j >= 0;j--)
			{
				if (dp[j]) //j之前的元素可以分成字典元素
				{
					if (dict.find(s.substr(j,i-j))!=dict.end())
					{
						dp[i] = true;
						break;
					}
				}
			}
		}

		return dp[s.size()];
	}
};

class Solution_139_new {
public:
	bool wordBreak(string s, vector<string>& wordDict) {

		set<string> strset(wordDict.begin(), wordDict.end());//vector-->set

		if (wordDict.size() == 0)
		{
			return false;
		}
		//动态规划
		vector<bool> dp(s.size() + 1, false);
		dp[0] = true;

		for (int i = 1; i <= s.size(); i++) //第一层遍历，s的每个位置是否可分成字典元素
		{
			for (int j = i - 1; j >= 0; j--)
			{
				if (dp[j]) //j之前的元素可以分成字典元素
				{
					if (strset.find(s.substr(j, i - j)) != strset.end()) //判断j-i元素是否match字典元素
					{
						dp[i] = true;
						break;
					}
				}
			}
		}

		return dp[s.size()];
	}
};

// world break -ii
class Solution_140 {

	vector<string> combine(string word, vector<string> prev){
		for (int i = 0; i < prev.size(); ++i){
			prev[i] += " " + word;
		}
		return prev;
	}

public:
	vector<string> wordBreak(string s, unordered_set<string>& dict) {

		vector<string> result;
		if (dict.count(s)){ //a whole string is a word
			result.push_back(s);
		}
		for (int i = 1; i < s.size(); ++i){
			string word = s.substr(i);
			if (dict.count(word)){
				string rem = s.substr(0, i);
				vector<string> prev = combine(word, wordBreak(rem, dict));
				result.insert(result.end(), prev.begin(), prev.end());
			}
		}

		reverse(result.begin(), result.end());
		return result;
	}
};

class Solution_140_ref{
	//运行时间：4ms
	//占用内存：508k

	vector<string> combine(string word, vector<string> prev){
		for (int i = 0; i < prev.size(); ++i){
			prev[i] += " " + word;
		}
		return prev;
	}

public:
	vector<string> wordBreak(string s, unordered_set<string>& dict) {
		
		vector<string> result;
		if (dict.count(s)){ //a whole string is a word
			result.push_back(s);
		}
		for (int i = 1; i < s.size(); ++i){
			string word = s.substr(i);
			if (dict.count(word)){
				string rem = s.substr(0, i);
				vector<string> prev = combine(word, wordBreak(rem, dict));
				result.insert(result.end(), prev.begin(), prev.end());       // result.begin()
			}
		}
		
		reverse(result.begin(), result.end());
		return result;
	}
};

// copy RandomListNode
class Solution_138 {
public:
	RandomListNode *copyRandomList(RandomListNode *head) {

		if (!head)
			return NULL;
		//if (!head->next) //bug: 这句话bug,不加才行！！！
		//{
		//	RandomListNode* copy = new RandomListNode(head->label);
		//	return copy;
		//}

		RandomListNode* cur = head;

		RandomListNode* temp = NULL;
		while (cur) //在每个节点后面copy节点
		{
			RandomListNode* newNode = new RandomListNode(cur->label); //bug 在之后插入

			temp = cur->next;
			cur->next = newNode;
			newNode->next = temp;
			
			cur = cur->next->next;
		}

		//每个节点copy random指针

		cur = head;
		while (cur) //bug 指针越界
		{
			if (cur->random)
			{
				cur->next->random = cur->random->next; //??->next
			}
			cur=cur->next->next;
		}

		// 将original list和copy list分开

		cur = head;
		RandomListNode* copyHead = new RandomListNode(0);
		RandomListNode* copy_cur = copyHead;

		RandomListNode* next = NULL;
		while (cur)
		{
			next = cur->next->next;

			copy_cur->next = cur->next;
			copy_cur = cur->next;

			cur->next = next;
			cur = next;

		}

		return copyHead->next;

		/*RandomListNode*p = NULL;
		p = head;
		RandomListNode* res = head->next;
		while (p->next != NULL)
		{
			RandomListNode* tmp = p->next;
			p->next = p->next->next;
			p = tmp;
		}
		return res;	*/
	}
};

// single number
class Solution_136 {
public:
	int singleNumber(int A[], int n) {
		//思路1：先排序，在相邻比较，时间O(nlogn)
		//思路2：异或运算，按二进制的位异或
		int ret = 0;
		for (unsigned int i = 0; i < n; i++)
		{
			ret ^= A[i];
		}
		return ret;
	}
};

//single number ii
class Solution_137 {
public:

	int singleNumber(int A[], int n) {
		int ret = 0;
		for (int i = 0; i < 32; i++)
		{
			int cnt = 0;
			for (int j = 0; j < n; j++)
			{
				cnt += (A[j] >> i) & 1;
			}
			ret += ((cnt % 3) << i);
		}
		return ret;
	}

	int singleNumber(vector<int>& nums) {
		//对每一位进行累加，对次数取模运算
		/* 把所有整数按照32位二进制进行每一位上的与1运算  结果为3n或3n+1;为3n+1的那些位就是只出现一次的数的二进制中1所在的位
		*/
		int ret = 0;
		for (int i = 0; i < 32; i++)
		{
			int cnt = 0;//计每一位的1的个数
			for (int j = 0; j < nums.size(); j++)
			{
				cnt += (nums[i] >> i) & 1; //0的不需要考虑
			}
			//把3n+1的那些位的1移回原位并累加起来 |=  也行  
			ret += (cnt % 3) << i;
		}
		return ret;
	}
};

// single nunmber iii
class Solution_260 {
public:
	vector<int> singleNumber(vector<int>& nums) {

		vector<int > vec;

		int result = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			result ^= nums[i]; //改变了nums[0]的值，后续使用有问题
		}

		//两个不同的数在不同的位上，若数字不同，则相应的位为1
		if (result == 0)
		{
			return vec;
		}

		//用flag找出num第一个不为0的位  
		int flag = 1;
		while ((flag&result) == 0)  // (flag&result) == 0
		{
			flag = flag << 1;
		}

		int data1 = 0, data2 = 0; //0 异或任何数是保留原值的作用
		for (int j = 0; j < nums.size(); j++)
		{
			if (nums[j] & flag) //某一位，相同的数与操作进同一分支
			{
				data1 ^= nums[j];
			}
			else
			{
				data2 ^= nums[j];
			}
		}
		vec.push_back(data1);
		vec.push_back(data2);

		return vec;
	}
};

// cany
class Solution_135 {

		//题意：N个孩子站成一排，每个孩子分配一个分值。给这些孩子派发糖果，满足如下要求：
		//每个孩子至少一个
		//分值更高的孩子比他的相邻位的孩子获得更多的糖果
		//求至少分发多少糖果？
public:
	int candy(vector<int> &ratings) {

		//采用左右遍历两次方法
		const int len = ratings.size();

		if (len==1)
		{
			return len;
		}
		vector<int> res(len,1);
		for (int i = 1; i < ratings.size();i++)
		{
			if (ratings[i]>ratings[i-1]) //右边大于左边
			{
				res[i] = res[i - 1] + 1; //分配的糖果数
			}
		}

		int ret = 0;
		for (int j = len - 2,ret=res[len-1]; j >= 0;j--) //bug  此处ret当做局部变量，
		{
			if (ratings[j]>ratings[j+1]&& res[j]<=res[j+1]) //左边大于右边且左边的糖果少于右边的糖果数
			{
				res[j] = res[j + 1] + 1; //或者其中较大者 mp[i]=max(mp[i],mp[i+1]+1);
			}
			ret += res[j];
		}

		return ret;
	}
};

//gas and cost canCompleteCircuit
class Solution_134 {
public:
	int canCompleteCircuit(vector<int> &gas, vector<int> &cost) {

		if (gas.size()==0||cost.size()==0||gas.size()!=cost.size())
		{
			return -1;
		}
		int index = 0;
		int sum_resdual = 0;

		for (int j = 0; j < gas.size();j++)
		{
			sum_resdual += (gas[j] - cost[j]);
		}
		if (sum_resdual<0)
		{
			return -1;
		}

		sum_resdual = 0;
		for (int i = 0; i < gas.size();i++)
		{
			//index = 0;  //
			sum_resdual += (gas[i] - cost[i]);
			if (sum_resdual<0)
			{
				index = i + 1;
				sum_resdual = 0;
			}
		}

		return index;
	}
};

// get maxProduct subArray
class Solution_152 {

	//分析：此题我们不仅要保存一个最大值，还得保存一个最小值，因为负负相乘等于正数，比如 - 2 4 - 5，即localMIn[i]表示以i结尾的连续字串乘积的最小值，而localMax[i]表示以i结尾的连续字串乘积的最大值，那么:
	//localMax[i] = max(max(localMin[i] * A[i], A[i] * localMax[i]), A[i]); 而localMin[i] = min(min(localMin[i] * A[i], A[i] * localMax[i]), A[i]);
public:
	int getMax(int x, int y)
	{
		return x > y ? x : y;
	}
	int getMin(int x, int y)
	{
		return x < y ? x : y;
	}
	int maxProduct(vector<int>& nums) {
		if (nums.size()==0)
		{
			return 0;
		}
		if (nums.size()==1)
		{
			return nums[0];
		}

		int local_min = nums[0];
		int local_max = nums[0];

		int global = nums[0];
		int local_max_copy = 0;
		for (size_t i = 1; i < nums.size(); i++)
		{
			local_max_copy = local_max;
			local_max = getMax(getMax(local_max*nums[i], nums[i]), local_min*nums[i]); //取三者中的大者
			local_min = getMin(getMin(local_min*nums[i], nums[i]), local_max*nums[i]);
			global = getMax(global, local_max);

		}

		return global;
	}
};

// clone graph
class Solution_133 {

	// date 2017/12/29 10:01
	// date 2017/12/29 11:04
public:

	UndirectedGraphNode *cloneGraph_r(UndirectedGraphNode *node) {
		unordered_map<UndirectedGraphNode*, UndirectedGraphNode*> hash;
		if (!node)
		{
			return node;
		}
		if (hash.find(node) != hash.end()) //找到，关键字已经访问过
		{
			hash[node] = new UndirectedGraphNode(node->label);
			for (auto iter : node->neighbors)
			{
				hash[node]->neighbors.push_back(cloneGraph(iter)); //递归DFS   //超时
			}
		}

		return hash[node];
	}


	UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node) //BFS
	{
		if (!node)
		{
			return node;
		}
		unordered_map<UndirectedGraphNode*, UndirectedGraphNode*> hash;
		UndirectedGraphNode* head = new UndirectedGraphNode(node->label);
		hash[node] = head;
		queue<UndirectedGraphNode*> que;
		que.push(node);  //que.push(head); bug 花费1小时查找

		while (!que.empty())
		{
			UndirectedGraphNode* q = que.front();
			que.pop();
			for (auto iter : q->neighbors)
			{
				if (!hash[iter]) //还没有访问
				{
					UndirectedGraphNode* temp = new UndirectedGraphNode(iter->label);
					hash[iter] = temp;
					que.push(iter);
				}

				hash[q]->neighbors.push_back(hash[iter]); //将一个节点的邻接点关系记录下来
			}

		}
		return hash[node];
	}

};

// Palindrome(回文) Partitioning
class Solution_131_ref {
	// date 2017/12/29 11:14

	// 如果要求输出所有可能的解，往往都是要用深度优先搜索。如果是要求找出最优的解，或者解的数量，往往可以使用动态规划
	// Reference：https://leetcode.com/problems/palindrome-partitioning/discuss/41964

public:
	vector<vector<string>> partition(string s) {
		vector<vector<string> > ret;
		if (s.empty()) return ret;

		vector<string> path;
		dfs(0, s, path, ret);

		return ret;
	}

	void dfs(int index, string& s, vector<string>& path, vector<vector<string> >& ret) {
		if (index == s.size()) {
			ret.push_back(path);
			return;
		}
		for (int i = index; i < s.size(); ++i) { //先以步长为1找回文串，找到了下一个回文串也是以步长为1开始 ；下一轮循环以步长为2开始，这样保证所有子回文串找到
			if (isPalindrome(s, index, i)) {
				path.push_back(s.substr(index, i - index + 1));
				dfs(i + 1, s, path, ret);
				path.pop_back();
			}
		}
	}

	bool isPalindrome(const string& s, int start, int end) {
		while (start <= end) {
			if (s[start++] != s[end--])
				return false;
		}
		return true;
	}

};

class Solution_131{

public:

	bool ispalindrome(string str)
	{
		if (str.size()==1)
		{
			return true;
		}

		bool flag = true;
		for (int i = 0; i < str.size()/2; i++)
		{
			if (str[i]==str[str.size()-1-i])
			{
				continue;
			}
			else
			{
				flag = false;
			}
		}
		return flag;
	}

	bool isPalindrome1(string s){
		return s == string(s.rbegin(), s.rend());
	}

	void dfs(string src, vector<string> &path, vector<vector<string>> &ret){
		if (src.size() <= 0 )
		{
			return; //递归退出条件
		}

		if (ispalindrome(src)) //最后部分的子串为回文串，结束此处递归 //
		{
			path.push_back(src);
			ret.push_back(path);
			path.pop_back();
		}

		string temp;
		for (int j = 1; j < src.size();j++)  //限制了长度2以上；
		{
			temp = src.substr(0, j);
		
			if (ispalindrome(temp))
			{
				path.push_back(temp);
				dfs(src.substr(j),path,ret);
				//ret.push_back(path);
				  
				path.pop_back();// 
			}

		}
		return;
	}

	void dfs1(string s, vector<string> &cur, vector<vector<string>> &res){
		if (s == ""){
			res.push_back(cur);
			return;
		}

		int len = s.size();
		for (int i = 1; i <= s.length(); ++i) {
			string sub = s.substr(0, i);
			if (ispalindrome(sub)){
				cur.push_back(sub);
				dfs1(s.substr(i, s.length() - i), cur, res);
				cur.pop_back();
			}
		}

	}

	vector<vector<string>> partition(string s) {

		vector<vector<string>> ret;
		vector<string > path;

		dfs1(s, path, ret); //深度递归

		return ret; 
	}
};

// palindrome partitioning ii
class Solution_132 {

	// 考虑是否要用回文检测（这里最优算法是不能专门写个函数用回文检测的，时间复杂度过高）
	// 其实可以在动态规划算法中直接检测是否是回文，不用专门写个函数
	//运行时间：6ms

	//占用内存：1416k

public:
	int minCut(string s) {

		int len = s.size();
		if (len<=1)
		{
			return 0;
		}
		vector<vector<int> > dp(len,vector<int>(len,0)); //表示i->j是否构成回文串
		vector<int> cnt(len + 1, INT_MAX);
		cnt[len + 1] = 0;

		for (int i = len - 1; i >= 0;i--)
		{
			for (int j = i; j < len;j++)
			{
				if (s[i]==s[j]&&(j-i<=1||dp[i+1][j-1]==1))
				{
					dp[i][j] = 1;
					cnt[i] = min(1 + cnt[j + 1], cnt[i]); // 最后一个INT_MAX+1溢出
				}
			}
		}
		return cnt[0] - 1;
	}
};

// Surrounded Regions
class Solution_130 {
public:
	void bfs(vector<vector<char>> &board, int cur_i, int cur_j, int row, int col)
	{
		if (board[cur_i][cur_j] != 'O')
			return;
		board[cur_i][cur_j] = '*';
		queue<pair<int, int >> q;
		q.push(make_pair(cur_i, cur_j));

		while (!q.empty())
		{
			int i = q.front().first;
			int j = q.front().second;
			q.pop();
			//board[i][j]='*';
			if (i - 1 >= 0 && board[i - 1][j] == 'O')
			{
				board[i - 1][j] = '*';
				q.push(make_pair(i - 1, j));
			}
			if (i + 1 < row&& board[i + 1][j] == 'O')    //i+1<=row bug:vector边界溢出，花了很长时间才找到！！！
			{
				board[i + 1][j] = '*';
				q.push(make_pair(i + 1, j));
			}
			if (j - 1 >= 0 && board[i][j - 1] == 'O')
			{
				board[i][j - 1] = '*';
				q.push(make_pair(i, j - 1));
			}
			if (j + 1 < col&&board[i][j + 1] == 'O')
			{
				board[i][j + 1] = '*';
				q.push(make_pair(i, j + 1));
			}
		}
	}

	void solve(vector<vector<char>> &board) {

		//bfs
		if (board.empty())
			return;
		int row = board.size();
		int col = board[0].size();
		if (row == 0 || col == 0)
			return;

		for (int i = 0; i < row; i++)
		{
			if (board[i][0] == 'O')
				bfs(board, i, 0, row, col);
			if (board[i][col - 1] == 'O')
				bfs(board, i, col - 1, row, col);
		}
		for (int j = 0; j < col; j++)
		{
			if (board[0][j] == 'O')
				bfs(board, 0, j, row, col);
			if (board[row - 1][j] == 'O')
				bfs(board, row - 1, j, row, col);
		}

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				board[i][j] = (board[i][j] == '*') ? 'O' : 'X';
			}
		}

	}
};

//Sum Root to Leaf Numbers
class Solution_129 {
public:
	int dfs(TreeNode* root, int sum)
	{
		if (!root)
		{
			return 0;
		}
		sum = sum * 10 + root->val;
		if (root->left==NULL&&root->right==NULL)
		{
			return sum;
		}

		return dfs(root->left, sum) + dfs(root->right, sum);
	}

	int sumNumbers_ref(TreeNode* root)
	{
		if (root == NULL)
		{
			return 0;
		}
		int sum = 0;

		return dfs(root, sum); //从根节点开始
	}

	int sumNumbers(TreeNode *root) {
		if (!root)
		{
			return 0;
		}
		stack<TreeNode*> sta;
		sta.push(root);

		int ret = 0;
		TreeNode* top = 0;
		while (!sta.empty())
		{
			top = sta.top();
			sta.pop();

			if (!top->left && !top->right)
			{
				ret += top->val;
			}
			if (top->left)
			{
				top->left->val += 10 * top->val; //但是这样改变了节点的值，可以将TreeNode* 和累积和(一个变量)组成一个pair，分开处理
				sta.push(top->left);
			}
			if (top->right)
			{
				top->right->val += 10 * top->val;
				sta.push(top->right);
			}
		}
		return ret;
	}
};

// Longest Consecutive Sequence
class Solution_128 {
public:
	int longestConsecutive(vector<int> &num) {
		if (num.size()==0)
		{
			return 0;
		}

		unordered_set<int> hash(num.begin(), num.end()); //O(n),插入hash表中

		int ret = 1;

		for (auto cur:num) //遍历元素O(n)
		{
			if (hash.find(cur)==hash.end()) //未找到当前元素
			{
				continue;
			}
			hash.erase(cur);

			int pre = cur - 1, next = cur + 1; //下一个连续序列重新赋值
			while (hash.find(pre)!=hash.end()) //hash的迭代器怎么实现？
			{
				hash.erase(pre);
				pre--;
			}
			while (hash.find(next)!=hash.end())
			{
				hash.erase(next);
				next++;
			}

			ret = max(ret, next - pre -  1); // bug : next - pre - 1
		}

		return ret;
	}
};

class Solution_127 {
public:
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		string start = beginWord;
		string end = endWord;
		unordered_set<string> dict(wordList.begin(), wordList.end());

		if (start == end || dict.empty())
		{
			return 0;
		}
		int ret = 1;

		queue<string> que; //bfs
		que.push(start);
		while (!que.empty())
		{
			int size = que.size();  //层次遍历，记录当前队列大小

			while (size--)  //不加这层循环，ret++是对每次元素出队列就计数；加循环就是严格每层计数++
			{
				string temp = que.front();
				que.pop();
				for (int i = 0; i < temp.size(); i++)
				{
					char ch = temp[i]; //对每个单词的每次字符替换构成新的单词，查看是否在字典中

					for (int j = 0; j < 26; j++) //对同一个单词处理，ret++不应该放在里面
					{
						temp[i] = 'a' + j; //替换
						if (dict.find(temp) != dict.end()) //找到了  需要吗？&&temp[i]!=ch不影响答案
						{
							//ret++; //找到一个计数+1 
							if (temp == end)
							{
								return ret + 1;  //找到答案，退出
							}
							que.push(temp);
							dict.erase(temp);
						}
					}
					temp[i] = ch; //还原单词
				}
			}
			ret++;
		}

		return 0;
	}
};

class Solution_127_old {

		//主要思想：广度优先搜索。先构造一个字符串队列，并将start加入队列。1.对队列头字符串做单个字符替换
		//每次替换后，2.判断是否和end匹配，如果匹配，返回答案；3.没有匹配，则在字典里面查询是否有“邻居字符串”,
		//如果有，则将该字符串加入队列，同时将该字符串从字典里删除。重复1的过程，知道和end匹配。如果最后队列
		//为空还未匹配到，则返回0.
public:
	int ladderLength(string start, string end, unordered_set<string> &dict) {
		if (start==end||dict.empty())
		{
			return 0;
		}
		int ret = 1;

		queue<string> que; //bfs
		que.push(start);
		while (!que.empty())
		{
			int size = que.size();  //层次遍历，记录当前队列大小

			while (size--)  //不加这层循环，ret++是对每次元素出队列就计数；加循环就是严格每层计数++
			{
				string temp = que.front();
				que.pop();
				for (int i = 0; i < temp.size(); i++)
				{
					char ch = temp[i]; //对每个单词的每次字符替换构成新的单词，查看是否在字典中

					for (int j = 0; j < 26; j++) //对同一个单词处理，ret++不应该放在里面
					{
						temp[i] = 'a' + j; //替换
						if (dict.find(temp) != dict.end()) //找到了  需要吗？&&temp[i]!=ch不影响答案
						{
							//ret++; //找到一个计数+1 
							if (temp == end)
							{
								return ret + 1;  //找到答案，退出
							}
							que.push(temp);
							dict.erase(temp);
						}
					}
					temp[i] = ch; //还原单词
				}
			}
			ret++; 
		}

		return 0;
	}
};

class Solution_126 {
public:
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {

		vector<vector<string>> res;
		unordered_set<string> visit;  //notice we need to clear visited word in list after finish this level of BFS
		queue<vector<string>> q;
		unordered_set<string> wordlist(wordList.begin(), wordList.end());

		q.push({ beginWord });
		bool flag = false; //to see if we find shortest path

		while (!q.empty()){
			int size = q.size();

			for (int i = 0; i < size; i++){            //for this level
				vector<string> cur = q.front();
				q.pop();
				vector<string> newadd = addWord(cur.back(), wordlist);
				for (int j = 0; j < newadd.size(); j++){   //add a word into path
					vector<string> newline(cur.begin(), cur.end());
					newline.push_back(newadd[j]);

					if (newadd[j] == endWord){
						flag = true;
						res.push_back(newline);
					}
					visit.insert(newadd[j]); // insert newadd word
					q.push(newline);
				}
			}

			if (flag) 
				break;  //do not BFS further 

			for (auto it = visit.begin(); it != visit.end(); it++) 
				wordlist.erase(*it); //erase visited one 
			visit.clear();
		}

		sort(res.begin(),res.end());
		return res;
	}

	// find words with one char different in dict
	// hot->[dot,lot]
	vector<string> addWord(string word, unordered_set<string>& wordlist){
		vector<string> res;
		for (int i = 0; i < word.size(); i++){
			char s = word[i];
			for (char c = 'a'; c <= 'z'; c++){
				word[i] = c;
				if (wordlist.count(word)) res.push_back(word);
			}
			word[i] = s;
		}
		return res;
	}
};

// isPalindrome
class Solution_125 {
public:
	bool isDigitandApha(char ch)  //isalpnum
	{
		if ((ch >= 'a'&&ch <= 'z') || (ch >= '0'&&ch <= '9'))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	bool isPalindrome(string s) {

		int size = s.size();
		int start = 0, end = size - 1;


		transform(s.begin(), s.end(), s.begin(), ::tolower);
		string copy = s;
		while (start <= end)
		{
			if (isDigitandApha(copy[start]) && isDigitandApha(copy[end]))
			{
				if (copy[start] == copy[end])
				{
					start++;
					end--;
				}
				else
				{
					return false;
				}
			}
			else if (!isDigitandApha(copy[start]))
			{
				start++;
			}
			else if (!isDigitandApha(copy[end]))
			{
				end--;
			}
			else //都不是字符或数字
			{
				start++;
				end--;
			}
		}

		return true;
	}

private:
	void test(char ch)
	{
		isalnum(ch);
		tolower(ch);
	}

};

// tree maxPathSum
class Solution_124 {

	int sum = INT_MIN;

	int help(TreeNode* root)
	{
		if (!root)
		{
			return 0;
		}
		int left = max(0, help(root->left));
		int right = max(0, help(root->right));

		sum = max(sum, left + right + root->val); //包括当前节点在路径上的最大值

		return max(left, right) + root->val; //返回以当前节点为根的路径最大值
	}

public:
	int maxPathSum(TreeNode *root) {  //考虑有负节点情况

		if (!root)
		{
			return 0;
		}

		help(root);

		return sum;
	}
};

//Best Time to Buy and Sell Stock
class Solution_121 {

	//分析一下价格曲线的不同走势对应的不同操作：
	//上升沿：此时都是收益的，我们需要判断现在的价格减去之前保留的最低的价格是否大于之前计算的最大收益即可。
	//下降沿：此时收益在降低，我们只需要判断新的价格是不是比之前保留的最低价格还要低即可。

	//1. 从后往前算， 因为卖肯定要比买之后，从后往前找到最大值，和当前值做差，差值的最大值就是获取的最大利润 
	//2. 找到第i天之前的最小值，然后prices[i] - min与最大收益maxprofit比较
	// 从前或者从后逆序都可以

public:
	int maxProfit_1(vector<int>& prices) {   //Time Limit Exceeded

		int ret = 0;

		for (int i = 0; i < prices.size() - 1; i++)
		{
			for (int j = i + 1; j < prices.size(); j++)
			{
				if (prices[j] - prices[i] >= 0)
				{
					ret = max(ret, prices[j] - prices[i]);
				}
			}
		}
		return ret;
	}

	int maxProfit(vector<int>& prices) {

		if (prices.size() <= 1)
		{
			return 0;
		}
		int max_profit = 0;
		int cur_profit = 0;
		int cur_min = prices[0];
		for (int i = 1; i < prices.size(); i++)
		{
			if (prices[i]<cur_min)
			{
				cur_min = prices[i]; //更新当前最小值
			}
			cur_profit = prices[i] - cur_min;
			if (cur_profit>max_profit)
			{
				max_profit = cur_profit;
			}
		}
		return max_profit;
	}
};

//Buy and Sell Stock ii
class Solution_122 {
	/*-本题由于允许多次交易（每次必须先卖出再买进），所以不好用爆搜
	- 分析可知，要想利益最大化，就应该每次在波谷买进，波峰卖出，这样利益最大，操作次数最少*/
public:
	int maxProfit(vector<int>& prices) {

		int max_profit = 0;
		for (int i = 1; i < prices.size();i++)
		{
			prices[i - 1] = prices[i] - prices[i - 1];
			if (prices[i-1]>0)
			{
				max_profit += prices[i - 1];
			}
		}

		return max_profit;
	}
	
};

//Buy and Sell Stock iii
class Solution_123 {
public:
	int maxProfit(vector<int> &prices) {

		//1.买一次相当于减一个数，
		//2.买卖两次维持当前最大的收益

		int buy1 = INT_MIN, sell1 = 0;
		int buy2 = INT_MIN, sell2 = 0;

		for (int i = 0; i < prices.size();i++)
		{
			buy1 = max(buy1, -prices[i]);
			sell1 = max(sell1, buy1 + prices[i]); //一次买卖现有的收益

			buy2 = max(buy2, sell1 - prices[i]); //又要使用一些钱
			sell2 = max(sell2, buy2 + prices[i]);
		}
		return sell2;
	}
};

//120. Triangle 
class Solution_120 {
public:
	// bottom-up
	int minimumTotal(vector<vector<int>>& triangle) {
		if (triangle.size() == 0)
			return 0;
		vector<int > vec(triangle.back());
		for (int i = triangle.size() - 2; i >= 0; i--)
		{
			for (int j = 0; j < triangle[i].size(); j++)
			{
				vec[j] = triangle[i][j] + min(vec[j], vec[j + 1]);
			}
		}
		return vec[0];
	}

	// top-down 
	int minimumTotal1(vector<vector<int>>& triangle) {
		vector<int> res(triangle.size(), triangle[0][0]);

		for (unsigned int i = 1; i < triangle.size(); i++)
		for (int j = i; j >= 0; j--) {
			if (j == 0)
				res[0] += triangle[i][j];
			else if (j == i)
				res[j] = triangle[i][j] + res[j - 1];
			else
				res[j] = triangle[i][j] + min(res[j - 1], res[j]);
		}
		return *min_element(res.begin(), res.end());
	}

	int minimumTotal2(vector<vector<int>>& triangle) {
		int row = triangle.size();

		if (row == 0)
		{
			return 0;
		}
		if (row == 1)
		{
			return triangle[0][0];
		}
		int ret = 0;


		vector<int> vec(triangle.size(), triangle[0][0]); //初始化

		for (int i = 1; i < row; i++) //当前行
		{
			for (int j = 0; j < triangle[i].size(); j++)
			{
				if (j == 0)
				{
					vec[j] = vec[j] + triangle[i][j];
				}
				else if (j == triangle[i].size() - 1)
				{
					vec[j] = vec[j - 1] + triangle[i][j]; //bug 会叠加上一次改变的值 //变顺序啊！！！逆序
				}
				else
				{
					vec[j] = triangle[i][j] + min(vec[j - 1], vec[j]);
				}
			}
		}
		return *min_element(vec.begin(), vec.end());
	}
};

// Pascal's Triangle 
class Solution_118 {
public:
	vector<vector<int>> generate(int numRows) {
		vector<vector<int>> vecs;
		vector<int> vec;
		if (numRows==0)
		{
			return vecs;
		}
		//if (numRows==1)  //放在循环中
		//{
		//	vec.push_back(1);
		//	vecs.push_back(vec);
		//	return vecs;
		//}
		//if (numRows == 2)
		//{
		//	vec.push_back(1);
		//	vecs.push_back(vec);
		//	vec.clear();
		//	vec.push_back(1);
		//	vec.push_back(1);
		//	vecs.push_back(vec);
		//	
		//	return vecs;
		//}
		//vec.push_back(1);
		//vecs.push_back(vec);
		//vec.clear();
		//vec.push_back(1);
		//vec.push_back(1);
		//vecs.push_back(vec);
		//vec.clear();
		for (int i = 1; i <= numRows;i++)
		{
			vec.resize(i,1);   //vector<int> tmp(i,1);
			for (int j = 1; j < vec.size()-1; j++)
			{
				vec[j] = vecs[i - 2][j - 1] + vecs[i - 2][j];
			}
			vecs.push_back(vec);
		}

		return vecs;
	}
};

// Pascal's Triangle II
class Solution_119 {
public:

	//  从后往前迭代
	vector<int> getRow1(int rowIndex) {

		vector<int> dp(rowIndex + 1, 1);

		for (int i = 2; i<rowIndex + 1; i++) {
			for (int j = i - 1; j>0; j--)
				dp[j] = dp[j] + dp[j - 1];

		}
		return dp;
	}

	vector<int> getRow(int rowIndex) {
		//A[i]=A[i-1]+A[i]      0<i<n-1
		vector<int> A;
		if (rowIndex < 0)  
			return A;
		A.resize(rowIndex + 1, 0);
		A[0] = 1; //第一行的数
		for (int k = 1; k<=rowIndex; k++){
			for (int j = k; j>0; j--){
				if (j == k)   
					A[j] = 1; //每行结尾的数
				else{
					A[j] = A[j] + A[j - 1];
				}
			}
		}
		return A;
	}
};

// Populating Next Right Pointers in Each Node
// 116-117用bfs代码一样
class Solution_116 {
public:
	//运行时间：8ms
    //占用内存：892k
	//使用层次遍历，每一层从左到右串接起来就行，每层最后一个元素next置NULL即可！
	void connect(TreeLinkNode *root) {
		if (!root)
		{
			return;
		}
		queue<TreeLinkNode*> que;
		que.push(root);

		while (!que.empty())
		{
			int size = que.size(); //每一层的大小

			TreeLinkNode* cur, *pre=NULL;
			while (size--)
			{
				cur= que.front();
				que.pop();

				if (pre)
				{
					pre->next = cur;		
				}
				pre = cur;
				if (cur->left)
				{
					que.push(cur->left);
				}
				if (cur->right)
				{
					que.push(cur->right);
				}
			}
			cur->next = NULL;

		}
	}

	void connect1(TreeLinkNode *root) {
		queue<TreeLinkNode*> q;
		if (!root)
			return;
		q.push(root);

		while (!q.empty()) {
			int size = q.size();
			for (int i = 0; i < size; i++) {
				TreeLinkNode* node = q.front();
				q.pop();
				if (i == size - 1)
					node->next = nullptr;
				else
					node->next = q.front();//当前节点已经出栈， q.front()为下一节点，避免记录上一次节点

				if (node->left)
					q.push(node->left);
				if (node->right)
					q.push(node->right);
			}
		}
	}
};

//115. Distinct Subsequences
class Solution_115 {
public:

	int numDistinct(string S, string T) { //母串和子串匹配的次数

		int lenx = T.size(); //子串
		int leny = S.size(); //母串
		if (lenx==0||leny==0)
		{
			return 0;
		}

		vector<vector<int> > dp(leny + 1, vector<int>(lenx + 1, 0));
		for (int i = 0; i <= leny;i++) //遍历母串
		{
			for (int j = 0; j <= lenx;j++) //遍历子串
			{
				if (j==0)
				{
					dp[i][j] = 1; //当子串长度为0时，所有次数都是1
					continue;
				}
				if (i>=1&&j>=1)
				{
					if (S[i - 1] == T[j - 1]) //当前母串和子串当前元素相等
					{
						dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
					}
					else
					{
						dp[i][j] = dp[i-1][j];
					}
				}
				
			}
		}
		return dp[leny][lenx];
	}

	int numDistinct2(string s, string t) {
		vector<int> match(t.size() + 1);
		match[0] = 1;
		for (int i = 0; i<s.size(); i++)
		for (int j = t.size(); j>0; j--)
			match[j] += (t[j - 1] == s[i] ? match[j - 1] : 0);
		return match[t.size()];
	}


	int numDistinct1(string S, string T) {  //bug.计算最长公共子序列
	//测试用例:
	//	"ddd", "dd"
	//	对应输出应该为:3
	//  你的输出为 : 2
		int lenx = S.size();
		int leny = T.size();
		if (lenx==0||leny==0)
		{
			return 0;
		}

		vector<vector<int>> vecs(leny+1, vector<int>(lenx+1, 0));
		for (int i = 1; i <= T.size();i++) //行
		{
			for (int j = 1; j <=lenx;j++) //列
			{
				if (S[j-1]==T[i-1])
				{
					vecs[i][j] = vecs[i-1][j - 1] + 1;
				}
				else
				{
					vecs[i][j] = max(vecs[i - 1][j], vecs[i][j - 1]);
				}
			}
		}

		int cnt = 0;
		if (vecs[leny][lenx] > 0){
			cnt++;
			for (int i = lenx - 1; i > 0; i--)
			{
				if (vecs[leny][i] == vecs[leny][lenx])
				{
					cnt++;
				}
			}
		}

		return cnt;
	}
};

//112. Path Sum 
class Solution_112 {
public:
	bool dfs(TreeNode* root, int cur_sum, int sum)
	{
		if (!root)
		{
			return false;
		}
		cur_sum += root->val;
		if (root->left == NULL&&root->right == NULL&&cur_sum == sum)
		{
			return true;
		}

		return dfs(root->left, cur_sum, sum) || dfs(root->right, cur_sum, sum);
	}

	bool hasPathSum(TreeNode *root, int sum) {
		if (!root)
		{
			return false;
		}
		return dfs(root, 0, sum);
	}


	bool hasPathSum1(TreeNode* root, int sum) { //递归
		if (root == nullptr){
			return false;
		}
		else if (root->left == nullptr && root->right == nullptr && root->val == sum){
			return true;
		}
		else{
			return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
		}
	}
	bool hasPathSum2(TreeNode *root, int sum) { //dfs

		//用非递归的先序和后序遍历感觉都有问题，需要额外的辅助变量才行
		if (!root)
		{
			return false;
		}
		stack<TreeNode*> sta;
		int ret = 0;

		TreeNode* p = root;
		TreeNode* temp;
		TreeNode* preNode;
		while (p!=NULL||!sta.empty())
		{
			while (p!=NULL)
			{
				sta.push(p);
				ret += p->val;
				p = p->left;
				if (p->left==NULL&&p->right==NULL)
				{
					if (ret==sum)
					{
						return true;
					}
				}
			}
			if (!sta.empty())
			{
				temp = sta.top(); //减有bug 
				if (temp->right&&preNode!=temp) //右孩子还未访问
				{
					p = p->right;  //到外层while
				}
				else
				{
					preNode = p;
					sta.pop();
					ret -= temp->val;
				}
			}
		}

		return false;
	}

	bool hasPathSum3(TreeNode* root, int sum) { //bfs
		if (!root)
		{
			return false;
		}

		queue<TreeNode*> que;
		queue<int> sum_que;
		que.push(root);
		sum_que.push(root->val);

		while (!que.empty())
		{
			TreeNode* cur = que.front();
			que.pop();
			int ret = sum_que.front();
			sum_que.pop();

			if (cur->left==NULL&&cur->right==NULL&&ret==sum)
			{
				return true;
			}
			if (cur->left)
			{
				que.push(cur->left);
				sum_que.push(ret + cur->left->val);
			}
			if (cur->right)
			{
				que.push(cur->right);
				sum_que.push(ret + cur->right->val);
			}
		}
		return false;

	}

};

//113. Path Sum II
class Solution_113 {
public:

	void dfs(TreeNode* root,int cur_sum,int sum,vector<int> &vec ,vector<vector<int>> &vecs)
	{
		if (!root)
		{
			return;
		}
	
		if (root->left==NULL&&root->right==NULL&&cur_sum==sum)
		{
			vecs.push_back(vec);
			return;
		}
		if (root->left)
		{
			vec.push_back(root->left->val);
			dfs(root->left, cur_sum + root->left->val, sum, vec, vecs);
			vec.pop_back();
		}
		if (root->right)
		{
			vec.push_back(root->right->val);
			dfs(root->right, cur_sum + root->right->val, sum, vec, vecs);
			vec.pop_back();
		}
		
		return;
	}

	vector<vector<int> > pathSum1(TreeNode *root, int sum) {
		vector<vector<int>> vecs;
		vector<int> vec;

		if (!root)
		{
			return vecs;
		}

		vec.push_back(root->val);
		dfs(root,root->val,sum,vec,vecs); //输入当前节点及其当前节点的和
		return vecs;
	}

	vector<vector<int> > pathSum(TreeNode *root, int sum) {

		vector<vector<int>> vecs;

		if (!root)
		{
			return vecs;
		}

		queue<TreeNode*> que;
		que.push(root);

		queue<vector<int>> path;
		path.push({ root->val }); 

		while (!que.empty())
		{
			TreeNode* temp;
			int size = que.size();

			for (int i = 0; i < size;i++)
			{
				temp = que.front();
				que.pop();

				vector<int>  vec= path.front();
				path.pop();

				if (temp->left==NULL&&temp->right==NULL&& accumulate(vec.begin(),vec.end(),0)==sum) //0 累加的初始值
				{
					vecs.push_back(vec);
				}
				if (temp->left)
				{
					que.push(temp->left);

					vector<int> var = vec;
					var.push_back(temp->left->val);
					path.push(var);
					//vec.pop_back();
				}
				if (temp->right)
				{
					que.push(temp->right);
					vector<int> var = vec;
					var.push_back(temp->right->val);
					path.push(var);
					//vec.pop_back();
				}
			}
		}
		return vecs;
	}
};

// is Balanced Binary Tree
class Solution_110 {
public:
	int getHeight(TreeNode* root)
	{
		if (!root)
		{
			return 0;
		}

		int left = getHeight(root->left);
		int right = getHeight(root->right);

		return (left > right) ? (left + 1) : (right + 1);
	}

	bool isBalanced(TreeNode *root) {
		if (!root)
		{
			return true;
		}
		// 不应该用<=1,return true,这样就提前返回了 
		//if (abs(getHeight(root->left) - getHeight(root->right)) > 1) {
		//	return false;
		//}
		//递归：自顶向下
		return  abs(getHeight(root->left) - getHeight(root->right) <= 1) && isBalanced(root->left) && isBalanced(root->right);
	}

	// method2
	bool isBalanced1(TreeNode *root) {
		int depth = 0;
		return isBalanced_helper(root, depth);
	}
	bool isBalanced_helper(TreeNode *root, int &depth) {
		if (root == NULL){
			depth = 0;
			return true;
		}
		int left, right;
		if (isBalanced_helper(root->left, left) && isBalanced_helper(root->right, right)){
			if (abs(left - right) <= 1){
				depth = 1 + max(left, right);
				return true;
			}
		}
		return false;
	}
};

// Convert Sorted Array to Binary Search Tree
class Solution_108 {
public:
	TreeNode* sortedArrayToBST(vector<int>& nums) {
		if (nums.size()==0)
		{
			return NULL;
		}
		if (nums.size()==1)
		{
			TreeNode* temp = new TreeNode(nums[0]);
			return temp;
		}

		int mid = nums.size() / 2;

		TreeNode* root = new TreeNode(nums[mid]);

		auto leftTree = vector<int>(nums.begin(), nums.begin() + mid);//最后一个迭代器指向最后一个元素的下一个位置
		auto rightTree = vector<int>(nums.begin() + mid + 1, nums.end());

		root->left = sortedArrayToBST(leftTree);
		root->right = sortedArrayToBST(rightTree);
		
		return root;
	}
};

// 109. Convert Sorted List to Binary Search Tree
class Solution_109 {
public:
	TreeNode *sortedListToBST(ListNode *head) {
		
		ListNode* slow = head;
		ListNode* fast = head;
		ListNode* pre = NULL;
		if (!head)
		{
			return NULL;
		}
		if (!head->next)
		{
			TreeNode* temp = new TreeNode(head->val);
			return temp;
		}
		while (fast&&fast->next)
		{
			pre = slow;
			slow = slow->next;
			fast = fast->next->next;
		}
		
		TreeNode* root = new TreeNode(slow->val);

		pre->next = NULL;
		slow = slow->next;
		root->left = sortedListToBST(head);
		root->right = sortedListToBST(slow);

		return root;
	}
};

// Binary Tree Level Order Traversal II
class Solution_107 {
public:
	//bfs
	vector<vector<int>> levelOrderBottom(TreeNode* root) {
		
		vector<vector<int> > vecs;
		stack<vector<int>> sta;
		if (!root)
		{
			return vecs;
		}

		queue<TreeNode*> que;
		que.push(root);

		while (!que.empty())
		{
			int size = que.size();
			vector<int> vec;
			TreeNode* temp;
			for (int i = 0; i < size;i++)
			{
				temp = que.front();
				que.pop();

				vec.push_back(temp->val);

				if (temp->left)
				{
					que.push(temp->left);
				}
				if (temp->right)
				{
					que.push(temp->right);
				}
			}
			sta.push(vec);
		}

		// reverse(res.begin(),res.end());
		while (!sta.empty())
		{
			vector<int> vec = sta.top();
			sta.pop();
			vecs.push_back(vec);
		}

		return vecs;
	}

    // 记录每层的个数；curCount当前层个数，nextCount下一层个节点个数
	vector<vector<int>> levelOrderBottom1(TreeNode* root) {
		int curCount = 0, nextCount = 0, level = 0;
		vector<vector<int>> ret;
		if (!root) return ret;

		curCount = 1;
		queue<TreeNode*> q;
		q.push(root);
		ret.push_back(vector<int>(0, curCount));

		while (!q.empty()) {
			if (curCount == 0) {
				curCount = nextCount;
				nextCount = 0;
				level++;
				ret.push_back(vector<int>(0, curCount)); //初始化下一层vector的大小
			}

			TreeNode *node = q.front();
			q.pop();
			curCount--;
			ret[level].push_back(node->val); //在同一层的节点加入

			if (node->left) {
				nextCount++;
				q.push(node->left);
			}
			if (node->right) {
				nextCount++;
				q.push(node->right);
			}
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}


	//dfs
	int getHeight(TreeNode *root)
	{
		if (!root) 
			return 0;
		return max(getHeight(root->left), getHeight(root->right)) + 1;
	}
	vector<vector<int> > levelOrderBottom2(TreeNode *root)
	{
		if (!root) 
			return vector<vector<int>>();

		vector<vector<int>> res(getHeight(root), vector<int>());

		dfs(root, res.size() - 1, res);
		return res;
	}
	void dfs(TreeNode *root, int height, vector<vector<int>> &res)
	{
		if (!root)
			return;
		res[height].push_back(root->val);

		dfs(root->left, height - 1, res);
		dfs(root->right, height - 1, res);
	}
};

// Construct Binary Tree from Preorder and Inorder Traversal
class Solution_105 {
public:

	//运行时间：9ms
	//占用内存：640k

	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {

		if (preorder.size()==0||inorder.size()==0||preorder.size()!=inorder.size())
		{
			return NULL;
		}

		TreeNode* root = new TreeNode(preorder[0]);

		if (preorder.size() == inorder.size() && inorder.size() == 1)
		{
			return root;
		}

		//auto pos = inorder.size() > 1 ? find(inorder.begin(), inorder.end(), preorder[0]) : inorder.begin();
		auto pos = find(inorder.begin(), inorder.end(), preorder[0]) ; 

		//preorder用下标分开也可以
		vector<int> inorder1(inorder.begin(), pos);  //pos指向容器最后一个元素的下一个位置
		vector<int> inorder2(pos + 1, inorder.end());

		vector<int> preorder1(preorder.begin() + 1, preorder.begin() + 1 + inorder1.size()); //cnt=inorder1.size()
		vector<int> preorder2(preorder.begin() + 1 + inorder1.size(), preorder.end());


		//auto iter = preorder.begin();
		//int cnt = 0;
		//while (iter != pos)
		//{
		//	iter++;
		//	cnt++;
		//}

		////preorder用下标分开也可以
		//vector<int> inorder1(inorder.begin(), pos);
		//vector<int> inorder2(pos + 1, inorder.end());
		//vector<int> preorder1(preorder.begin() + 1, preorder.begin() + 1 + cnt); //cnt=inorder1.size() //报alloc错误
		//vector<int> preorder2(preorder.begin() + 1 + cnt, preorder.end());

		if (preorder1.size()>0)
		{
			root->left = buildTree(preorder1, inorder1);
		}
		
		if (preorder2.size()>0)
		{
			root->right = buildTree(preorder2, inorder2);
		}
		
		return root;
	}


	TreeNode *buildTree1(vector<int> &preorder, vector<int> &inorder) {
		return build(preorder, inorder, 0, preorder.size() - 1, 0, inorder.size() - 1);
	}
	TreeNode *build(vector<int> &preorder, vector<int> &inorder, int l1, int r1, int l2, int r2)
	{
		if (l1 > r1)
			return NULL;
		int gen = preorder[l1];
		int i, cnt = 0;

		for (i = l2; i <= r2&&inorder[i] != gen; cnt++, i++); //找到当前根节点在inorder中的位置

		TreeNode *root = (TreeNode *)malloc(sizeof(TreeNode));
		root->val = gen;
		root->left = build(preorder, inorder, l1 + 1, l1 + cnt, l2, i - 1); //位置信息要准确
		root->right = build(preorder, inorder, l1 + 1 + cnt, r1, i + 1, r2);
		return root;
	}


public:
	using iter = std::vector<int>::iterator;
public:
	TreeNode* buildTree2(vector<int>& inorder, vector<int>& postorder)
	{
		return buildTreeHelper(inorder.begin(), inorder.end(), postorder.begin(), postorder.end());
	}

	TreeNode* buildTreeHelper(iter inOrderBegin, iter inOrderEnd, iter postOrderBegin, iter postOrderEnd)
	{
		if (inOrderBegin == inOrderEnd)
			return nullptr;
		if (std::next(inOrderBegin) == inOrderEnd)
			return new TreeNode(*inOrderBegin);
		TreeNode *root = new TreeNode(*std::prev(postOrderEnd));
		auto pivot = std::find(inOrderBegin, inOrderEnd, root->val);
		auto leftSize = std::distance(inOrderBegin, pivot);
		auto rightSize = std::distance(pivot, inOrderEnd) - 1;
		if (leftSize != 0)
			root->left = buildTreeHelper(inOrderBegin, pivot, postOrderBegin, std::next(postOrderBegin, leftSize));
		if (rightSize != 0)
			root->right = buildTreeHelper(std::next(pivot), inOrderEnd, std::next(postOrderBegin, leftSize), std::prev(postOrderEnd));
		return root;
	}
};

// 106. Construct Binary Tree from Inorder and Postorder Traversal
class Solution_106 {
public:
	TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) { //这样消耗内存多些
		
		if (inorder.size()==0||postorder.size()==0||inorder.size()!=postorder.size())
		{
			return NULL;
		}

		int len = postorder.size();
		TreeNode* root = new TreeNode(postorder[len-1]);
		
		//bug:
		//terminate called after throwing an instance of 'std::bad_alloc'
		//what() : std::bad_alloc

		auto pos = find(inorder.begin(), inorder.end(), postorder[len - 1]);
		vector<int> inorder_l(inorder.begin(),pos);
		vector<int> inorder_r(pos + 1, inorder.end());
		vector<int> postorder_l(postorder.begin(), postorder.begin() + inorder_l.size());
		vector<int> postorder_r(postorder.begin()+inorder_l.size(),postorder.end()-1);
		
		if (inorder_l.size()>0)
		{
			root->left = buildTree(inorder_l, postorder_l);
		}

		if (inorder_r.size()>0)
		{
			root->right = buildTree(inorder_r, postorder_r);
		}
		
		return root;
	}


public:
	TreeNode* buildTreeHelper(vector<int>& inorder, int l1, int r1, vector<int>& postorder, int l2, int r2) //在原数组上操作，不需要额外空间
	{
		if (l1>r1||l2>r2)
		{
			return NULL;
		}
		
		TreeNode* root = new TreeNode(postorder[r2]);
		int i = 0;
		for ( i= l1; i <= r1;i++) // for ( i= 0; i < inorder.size();i++) //递归实现参数要能进入下次递归
		{
			if (inorder[i]==postorder[r2])
			{
				break;
			}
		}

		root->left = buildTreeHelper(inorder,l1,i-1,postorder,l2,l2+(i-1-l1)); //慢慢体会下标的准确性
		root->right = buildTreeHelper(inorder, i + 1, r1, postorder, l2 + (i - l1), r2-1);

		return root;
	}

	TreeNode* buildTree2(vector<int>& inorder, vector<int>& postorder) 
	{
		if (inorder.size()==0||postorder.size()==0||inorder.size()!=postorder.size())
		{
			return NULL;
		}

		return buildTreeHelper(inorder, 0, inorder.size() - 1, postorder,0, postorder.size() - 1);
	}
};

// Maximum Depth of Binary Tree
class Solution_104 {
public:

	// Time Limit Exceeded
	int maxDepth1(TreeNode *root) {

		if (!root)
		{
			return	0;
		}

		return (maxDepth1(root->left) > maxDepth1(root->right)) ? (maxDepth1(root->left) + 1) : (maxDepth1(root->right) + 1);
	}

	// 这样写不会超时
	int maxDepth(TreeNode* root) {
		if (root == NULL) 
			return 0;
		return max(maxDepth(root->left), maxDepth(root->right)) + 1;
	}

	int maxDepth2(TreeNode *root)
	{
		if (!root)
		{
			return 0;
		}
		queue<TreeNode*> que;
		que.push(root);

		int level = 0;
		while (!que.empty())
		{
			int size = que.size();
			for (int i = 0; i < size;i++)
			{
				TreeNode* temp = que.front();
				que.pop();

				if (temp->left)
				{
					que.push(temp->left);
				}
				if (temp->right)
				{
					que.push(temp->right);
				}
			}
			level++;
		}

		return level;
	}
};

// 102. Binary Tree Level Order Traversal
class Solution_102 {
public:
	vector<vector<int> > levelOrder(TreeNode *root) {

		vector<vector<int>> vecs;

		if (!root)
		{
			return vecs;
		}
		queue<TreeNode*> que;
		que.push(root);

		while (!que.empty())
		{
			int size = que.size();
			vector<int> vec;

			while (size--)
			{
				TreeNode* temp = que.front();
				que.pop();

				vec.push_back(temp->val);
				if (temp->left)
				{
					que.push(temp->left);
				}
				if (temp->right)
				{
					que.push(temp->right);
				}
			}
			vecs.push_back(vec);
		}

		return vecs;
	}

	vector<vector<int>> levelOrder1(TreeNode* root) {
		if (!root) { return{}; }
		vector<int> row;
		vector<vector<int> > result;
		queue<TreeNode*> q;
		q.push(root);
		int count = 1;

		while (!q.empty()) {
			if (q.front()->left)
			{
				q.push(q.front()->left);
			}
			if (q.front()->right)
			{
				q.push(q.front()->right);
			}
			row.push_back(q.front()->val);
			q.pop();
			if (--count == 0) {
				result.emplace_back(row);
				row.clear();
				count = q.size();
			}
		}
		return result;
	}

	// 递归实现
	vector<vector<int>> ret;
	void buildVector(TreeNode *root, int depth)
	{
		if (root == NULL) return;
		if (ret.size() == depth)
			ret.push_back(vector<int>());

		ret[depth].push_back(root->val);
		buildVector(root->left, depth + 1);
		buildVector(root->right, depth + 1);
	}

	vector<vector<int> > levelOrder2(TreeNode *root) {
		buildVector(root, 0);
		return ret;
	}
};

// 103. Binary Tree Zigzag Level Order Traversal
class Solution_103 {
public:
	vector<vector<int> > zigzagLevelOrder(TreeNode *root) {

		vector<vector<int> > vecs;
		if (!root)
		{
			return vecs;
		}

		queue<TreeNode*> que;
		que.push(root);

		int level = 0;
		while (!que.empty())
		{
			int size = que.size();
			vector<int> vec;
			while (size--)
			{
				TreeNode* temp = que.front();
				que.pop();
				vec.push_back(temp->val);
				if (temp->left)
				{
					que.push(temp->left);
				}
				if (temp->right)
				{
					que.push(temp->right);
				}
			}
			level++;
			if (!(level % 2))  //实际是偶数行反转，从第0行开始       //奇数层反转vector  bug:level % 2 != 0 
			{
				reverse(vec.begin(), vec.end());
			}
			vecs.push_back(vec);
		}

		return vecs;
	}
};

// 101. Symmetric Tree
class Solution_101 {
public:

	// 递归调用，同时判断左子树的左节点与右子树的右节点，以及左子树的右节点与右子树的左节点。一旦这两个节点不相等，就返回false。 
	bool isSymmetricHelp(TreeNode *left, TreeNode* right)
	{
		if (!left&&!right)
		{
			return true;
		}
		if (!left&&right ||left&&!right)
		{
			return false;
		}
		if (left->val!=right->val)
		{
			return false;
		}

		return isSymmetricHelp(left->left, right->right) && isSymmetricHelp(left->right, right->left);
	}

	bool isSymmetric(TreeNode *root) {

		if (!root)
		{
			return true;
		}
		return isSymmetricHelp(root->left, root->right);
	}

	//非递归实现
	//需要：对每一层成对送入队列，出队列比较
	bool isSymmetric1(TreeNode* root) {
		if (!root)
		{
			return true;
		}
		queue<TreeNode*> que;

		que.push(root->left);
		que.push(root->right);

		while (!que.empty())
		{
			int size = que.size();
			while (size)
			{
				TreeNode* left = que.front();
				que.pop();
				TreeNode* right = que.front(); //取出成对的元素
				que.pop();
				size -= 2;
				if (!left&&!right)
				{
					continue;
				}
				if (!left&&right)
				{
					return false;
				}
				if (left&&!right)
				{
					return false;
				}

				if (left->val!=right->val)
				{
					return false;
				}
				que.push(left->left); //非递归实现时：有NULL节点也入栈或者队列
				que.push(right->right);
				que.push(left->right);
				que.push(right->left);
			}
		}
		return true;
	}
};

// same tree
class Solution_100 {
public:
	bool isSameTree(TreeNode *p, TreeNode *q) {

		if (!p&&!q)
		{
			return true;
		}
		if (!p&&q)
		{
			return false;
		}
		if (p&&!q)
		{
			return false;
		}
		if (p->val!=q->val)
		{
			return false;
		}

		return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
	}

	//非递归实现时：有NULL节点也入栈或者队列
	// ref：https://leetcode.com/problems/same-tree/discuss/32683
};

// add two sum
class Solution_1 {
public:
	// O(n^2)
	vector<int> twoSum(vector<int> &numbers, int target) {

		vector<int> vec;
		if (numbers.size()==0)
		{
			return vec;
		}

		for (int i = 0; i < numbers.size()-1;i++)
		{
			for (int j = i + 1; j < numbers.size();j++)
			{
				if (numbers[i]+numbers[j]==target)
				{
					vec.push_back(i);
					vec.push_back(j);
					break;
				}
			}
		}

		return vec;
	}

	// 使用一个哈希表来解，第一遍扫描，保存到哈希表中，第二遍扫，看target-n在不在哈希表中，时间复杂度为O(n)。
	vector<int> twoSum(vector<int> a, int target) {
		int i, j, k, l, m, n;
		map<int, int>mymap;
		map<int, int>::iterator it;
		vector<int>ans;
		for (i = 0; i < a.size(); i++){
			it = mymap.find(target - a[i]);
			if (it != mymap.end()){
				ans.push_back(it->second);
				ans.push_back(i);
				return ans;
			}
			else{
				mymap.insert(make_pair(a[i], i));
			}
		}
	}

	vector<int> twoSum1(vector<int>& nums, int target) {
		vector<int> vec;
		if (nums.size()==0)
		{
			return vec;
		}
		sort(nums.begin(), nums.end()); 
		int start = 0, end = nums.size()-1;
		while (start<end)
		{
			if (nums[start]+nums[end]==target)
			{
				vec.push_back(start);
				vec.push_back(end);
				break;
			}
			else if (nums[start] + nums[end] < target)
			{
				start++;
			}
			else
			{
				end--;
			}
		}
		return vec; //返回值的排序后的index,不符合题意
	}
};

// addTwoNumbers link
class Solution_2 {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		if (!l1)
		{
			return l2;
		}
		if (!l2)
		{
			return l1;
		}

		ListNode* head = new ListNode(0),*pre=head;
		ListNode* cur = NULL;
		bool flag = false; //进位标志

		int sum = 0;
		while (l1!=NULL&&l2!=NULL)
		{
			sum = l1->val + l2->val;
			if (flag)
			{
				sum = sum + 1;
				flag = false;
			}		
			if (sum>=10)
			{
				sum = sum % 10;
				flag = true;
			}
			cur = new ListNode(sum);
			pre->next=cur;
			pre = pre->next;
			l1 = l1->next;
			l2 = l2->next;
		}

		while (l2 != NULL)
		{
			if (flag)
			{
				sum = l2->val + 1;
				flag = false;
			}
			if (sum >= 10)
			{
				sum = sum % 10;
				flag = true;
			}
			cur = new ListNode(sum);
			pre->next = cur;
			pre = pre->next;
			l2 = l2->next;
		}
		while (l1 != NULL)
		{
			if (flag)
			{
				sum = l1->val + 1;
				flag = false;
			}
			if (sum >= 10)
			{
				sum = sum % 10;
				flag = true;
			}
			cur = new ListNode(sum);
			pre->next = cur;
			pre = pre->next;
			l1 = l1->next;
		}
		if (flag)
		{
			cur = new ListNode(1);
			pre->next = cur;
		}
		return head->next;
	}
};

// 3. Longest Substring Without Repeating Characters
class Solution_3 {
public:
	int lengthOfLongestSubstring(string s) {

		if (s.size()==0)
		{
			return 0;
		}

		int ret = 0;
		map<char, int> mp;
		int i = 0;
		for (int j = 0; j < s.size();j++)
		{
			if (mp.count(s[j])) //找到了,没有插入当前元素，之前插入的相同元素充当当前元素,但要覆盖当前元素的位置second，//或者覆盖
			{
				ret = max(ret, j - i);

				if (mp[s[j]]>=i) //有可能返回去 "abba"
				{
					i = mp[s[j]] + 1; //下次从第一次个重复的下一个位置开始
				}

				mp[s[j]] = j; //更新位置
				
			}
			else
			{
				mp.insert(make_pair(s[j],j));
			}
		}
		ret = ret > (s.size() - i) ? ret : (s.size() - i); // 尾处理

		return ret;
	}
};

// 4. Median of Two Sorted Arrays
class Solution_4 {
public:
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {

	}
};

// 5. Longest Palindromic Substring
class Solution_5 {
public:

	// reverse(res.begin(),res.end());
	// return s == string(s.rbegin(), s.rend());
	bool isPalindrome(string s)
	{
		return s == string(s.rbegin(), s.rend());
	}

	// 处理bb和aba的情况
	string expandstring(const string s, int l, int r)
	{
		while (l >= 0 && r < s.size() && s[l] == s[r])
		{
			l--;
			r++;
		}

		return s.substr(l + 1, r - l - 1);
	}

	string longestPalindrome(string s) {
		if (s.size() <= 1)
		{
			return s;
		}
		string longestr = s.substr(0, 1);
		for (int i = 0; i < s.size() - 1; i++)
		{
			string expand = expandstring(s, i, i);
			if (expand.length()>longestr.size())
			{
				longestr = expand;
			}
			string expand2 = expandstring(s, i, i + 1);
			if (expand2.length() > longestr.length())
			{
				longestr = expand2;
			}
		}
		return longestr;
	}
};

// 6. ZigZag Conversion
class Solution_6 {
public:
	string convert(string s, int numRows) {
		if (numRows <= 1)
		{
			return s;
		}
		//vector<string> vec; //numRows行 //vector<string> zigzag(nRows,"");要初始化行数，否则不能使用
		//string vec[numRows]; bug

		string* vec = new string[numRows];
		int step = 0, row = 0;
		for (int i = 0; i < s.size(); ++i)
		{
			vec[row].push_back(s[i]);
			if (row == 0)
			{
				step = 1;
			}
			if (row == numRows - 1)
			{
				step = -1;
			}
			row = row + step;
		}

		string str = "";
		for (int i = 0; i < numRows; i++)
		{
			str += vec[i];
		}
		delete[] vec;

		return str;
	}
};

//8. String to Integer (atoi)
class Solution_8 {
public:
	int atoi1(const char *str) {
		atoi(str);
	}

	int myAtoi(string str) { //int float

		long ret = 0;
		if (str.size() == 0)
		{
			return 0;
		}
		int i = 0;
		while (i < str.size() && isspace(str[i]))
		{
			i++;
		}
		bool flag = false;
		if (i < str.size() && (str[i] == '-' || str[i] == '+'))
		{
			if (str[i] == '-')
				flag = true;
			i++;
		}
		if (str[i] == '+' || str[i] == '-')  //多个正负号，不合法 +-2
		{
			return 0;
		}

		while (i < str.size() && str[i] == '0') //  -0012a42
		{
			i++;
		}
		while (i < str.size())
		{
			if (str[i] >= '0'&&str[i] <= '9')
			{
				ret = ret * 10 + str[i] - '0';

				long temp = flag ? ret*(-1) : ret; //中间结果就判断
				if (temp > INT_MAX)
				{
					return (INT_MAX);
				}
				if (temp < INT_MIN)
				{
					return INT_MIN;
				}

				i++;
			}
			else
			{
				break;  //-0012a42 截断a 
			}

		}
		return flag ? ret*(-1) : ret; // 
	}
};

// 9. Palindrome Number
class Solution_9 {
public:
	bool isPalindrome(int x) { //+-处理
		if (x<0)
		{
			return false; //负数不当做回文
		}
		if (x%10==0&&x!=0)
		{
			return false;
		}
		if (x<10)
		{
			return true;
		}
		int invertnum = 0;
		while (x>invertnum)  //Revert half of the number
		{
			invertnum = invertnum * 10 + x % 10;
			x /= 10;
		}

		return x == invertnum||x==invertnum/10; //121 ||1221
	}
};

// 10. Regular Expression Matching
class Solution_10 {
public:
	/*
	动态规划
	如果 p[j] == str[i] || pattern[j] == '.', 此时dp[i][j] = dp[i-1][j-1];
	如果 p[j] == '*'
	分两种情况:
	1: 如果p[j-1] != str[i] && p[j-1] != '.', 此时dp[i][j] = dp[i][j-2] //*前面字符匹配0次
	2: 如果p[j-1] == str[i] || p[j-1] == '.'
	此时dp[i][j] = dp[i][j-2] // *前面字符匹配0次
	或者 dp[i][j] = dp[i][j-1] // *前面字符匹配1次
	或者 dp[i][j] = dp[i-1][j] // *前面字符匹配多次
	*/
	bool isMatch(string s, string p) { //p去匹配s

		vector<vector<bool> > dp(s.size() + 1, vector<bool>(p.size() + 1, false));

		dp[0][0] = true; // 空串匹配空串
		//第一列空串p去匹配，为false
		//第一行非空串p去匹配空串s;只要p中有*，就可以匹配
		for (int i = 1; i < dp[0].size(); i++)
		{
			if (p[i - 1] == '*')
			{
				dp[0][i] = i>1 && dp[0][i - 2];
			}	
		}
		for (int i = 1; i <= s.size();i++)
		{
			for (int j = 1; j <= p.size();j++)
			{
				if (s[i-1]==p[j-1]||p[j-1]=='.') //直接匹配成功
				{
					dp[i][j] = dp[i - 1][j - 1];
				}
				else if (p[j-1]=='*')
				{
					if (s[i-1]!=p[j-2]&&p[j-2]!='.') //匹配*前面的字符0次,跳过当前p
					{
						dp[i][j] = dp[i][j-2];
					}
					else
					{
					    //*前面字符匹配1次 || *前面字符匹配0次 || *前面字符匹配多次
						dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
					}
				}
			}
		}

		return dp[s.size()][p.size()];
	}
};

// 11. Container With Most Water 
class Solution_11 {
public:
	int maxArea(vector<int> &height) {
		int start = 0;
		int end = height.size() - 1;

		int max_area = 0;
		while (start<end)
		{
			max_area = max(max_area, min(height[start],height[end])*(end-start) );
			if (height[start]<=height[end])
			{
				start++;
			}
			else
			{
				end--;
			}
		}
		return max_area;
	}
};

// intToRoman
class Solution_12 {
public:
	string intToRoman(int num) {
		char* c[4][10] = {
			{ "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" },
			{ "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" },
			{ "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" },
			{ "", "M", "MM", "MMM" }
		}; //指针数组
		string roman;
		roman.append(c[3][num / 1000 % 10]);
		roman.append(c[2][num / 100 % 10]);
		roman.append(c[1][num / 10 % 10]);
		roman.append(c[0][num % 10]);

		return roman;
	}

	string intToRoman1(int num) {
		string str;
		string symbol[] = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
		int value[] = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
		for (int i = 0; num != 0; ++i)
		{
			while (num >= value[i])
			{
				num -= value[i];
				str += symbol[i];
			}
		}
		return str;
	}
};

class Solution_13 {
public:
	int romanToInt(string s) { // "DCXXI"  "MCMXCVI"
		int ret = 0;
		unordered_map<char, int> mp{ { 'I', 1 }, { 'V', 5 }, { 'X', 10 }, { 'L', 50 }, { 'C', 100 }, { 'D', 500 }, { 'M', 1000 } };
		ret = mp[s[s.size()-1]];
		for (int i = s.size()-2; i >=0; --i)
		{
			if (mp[s[i]] >= mp[s[i+1]]) // 从后往前走
			{
				ret += mp[s[i]];
			}
			else
			{
				ret -= mp[s[i]]; //"CM":100-100=900
			}
		}
		return ret;
	}
};

// longestCommonPrefix
class Solution_14 {
public:
	string longestCommonPrefix(vector<string> &strs) {
		if (strs.size() == 0)
		{
			return ""; //返回NULL 错误
		}
		if (strs.size() == 1)
		{
			string str = strs[0];
			return str;
		}
		string ret = strs[0];
		int max_comlen = ret.size();
		int j = 0;
		for (int i = 1; i < strs.size(); i++)
		{
			j = 0;
			while (strs[i][j] == ret[j] && j < ret.size()) //优化：记录上一次匹配最远的位置，下一次不超过这个距离
			{
				j++;
			}
			max_comlen = min(max_comlen, j); //取的是最小长度的公共子串；min=('a','a','b')=0;
		}
		return ret.substr(0, max_comlen);
	}
};

// threeSum
class Solution_15 {
public:

	void twosum(vector<vector<int>>& vecs, vector<int>& nums, int start, int target)
	{
		vector<int> ans;
		int end = nums.size() - 1;
		while (start < end)
		{
			if (nums[start] + nums[end] == target)
			{
				ans.push_back(-target);
				ans.push_back(nums[start]);
				ans.push_back(nums[end]);
				vecs.push_back(ans);
				ans.clear();
				//start++; end--;  跳不过重复的元素
				while (start < end&&nums[start] == nums[start + 1])
				{
					start++;
				}
				while (start < end&&nums[end] == nums[end - 1])
				{
					end--;
				}
				start++; end--;
			}
			else if (nums[start] + nums[end] < target)
			{
				start++;
			}
			else
			{
				end--;
			}
		}
		return;
	}

	vector<vector<int>> threeSum(vector<int>& nums) { // Time Limit Exceeded
		vector<vector<int>> vecs;
		if (nums.size() <= 2)
		{
			return vecs;
		}
		sort(nums.begin(), nums.end());
		for (int i = 0; i < nums.size() - 2; ++i)
		{
			if (i>0 && nums[i] == nums[i - 1]) //忽略掉有重复元素的值//忽略掉后面重复的元素
			{
				continue;
			}
			twosum(vecs, nums, i + 1, -nums[i]);
		}
		return vecs; //
	}

	vector<vector<int>> threeSum1(vector<int>& nums) { // Time Limit Exceeded
		vector<vector<int>> vecs;
		if (nums.size() <= 2)
		{
			return vecs;
		}

		unordered_map<int, int> mp;
		vector<int> ans;
		for (int i = 0; i < nums.size() - 2; ++i)
		{
			mp.clear();
			for (int j = i + 1; j < nums.size(); ++j)
			{
				auto iter = mp.find(-(nums[i] + nums[j])); //查找主关键字
				if (iter != mp.end())
				{
					ans.push_back(nums[i]);
					ans.push_back(iter->first);
					ans.push_back(nums[j]);
					sort(ans.begin(), ans.end());  //处理有重复元素
					if (find(vecs.begin(), vecs.end(), ans) == vecs.end()) //没有元素才插入操作
					{
						vecs.push_back(ans);
					}
					ans.clear();
				}
				else
				{
					mp.insert(make_pair(nums[j], j));
				}
			}
		}
		return vecs; //
	}

};

// three sum closest
class Solution_16 {
public:
	// 三个指针操作O(N^2) test=（[0,2,1,-3] 1）;([1,1,-1,-1,3]-1)
	int threeSumClosest(vector<int>& nums, int target) {
		if (nums.size() < 3)
		{
			return 0;
		}

		int closest_sum = nums[0] + nums[1] + nums[2];
		sort(nums.begin(), nums.end());
		for (int first = 0; first < nums.size() - 2; first++)
		{
			if (first > 0 && nums[first] == nums[first - 1])
			{
				continue;
			}
			int second = first + 1;
			int end = nums.size() - 1;
			while (second<end)
			{
				int temp = nums[first] + nums[second] + nums[end];

				if (abs(closest_sum - target)>abs(temp - target))
				{
					closest_sum = temp;
				}
				if (temp == target)
				{
					return temp;
				}
				else if (temp < target)
				{
					second++;
				}
				else
				{
					end--;
				}
			}
		}
		return closest_sum;
	}

	// 暴力O(n^3)
	int threeSumClosest_1(vector<int> &num, int target)
	{
		int res = num[0] + num[1] + num[2];
		for (int i = 0; i < num.size() - 2; i++){
			for (int j = i + 1; j < num.size() - 1; j++){
				for (int t = j + 1; t < num.size(); t++){
					int tem = num[i] + num[j] + num[t];
					if (abs(target - tem) < abs(target - res))
						res = tem;
					if (res == target)
						return res;
				}
			}
		}
		return res;
	}
};

// 17. Letter Combinations of a Phone Number
class Solution_17 {
public:

	// map<int, string> num2alp = { { 2, "abc" }, { 3, "def" }, { 4, "ghi" }, { 5, "jkl" }, { 6, "mno" }, { 7, "pqrs" }, { 8, "tuv" }, { 9, "wxyz" } };
	
	void help(string digits,vector<string> &vec,string &temp,int index,int n)
	{
		static const vector<string> v = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		if (index==n)
		{
			vec.push_back(temp);
			return;
		}
		string str = v[digits[index]-'0'];
		
		for (int i = 0; i < str.size();i++)
		{
			temp.push_back(str[i]);
			help(digits, vec,temp, index + 1,n);
			temp.pop_back();
		}
		return;
	}

	// 回溯递归实现
	vector<string> letterCombinations(string digits) { //23
		vector<string> vec;
		string temp;
		
		if (digits.size()==0)
		{
			vec.push_back(""); //初始化一个空
			return vec;
		}

		help(digits,vec,temp,0,digits.size());

		return vec;
	}

	// 迭代实现
	vector<string> letterCombinations_ref(string digits) {
		vector<string> result;
		if (digits.empty()) return vector<string>();
		static const vector<string> v = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		result.push_back("");   // add a seed for the initial case
		for (int i = 0; i < digits.size(); ++i) {
			int num = digits[i] - '0';
			if (num < 0 || num > 9) break;
			const string& candidate = v[num];
			if (candidate.empty()) continue;
			vector<string> tmp;
			for (int j = 0; j < candidate.size(); ++j) {
				for (int k = 0; k < result.size(); ++k) {
					tmp.push_back(result[k] + candidate[j]);
				}
			}
			result.swap(tmp); //
		}
		return result;
	}

};

// 4sum
class Solution_18 {
public:
	vector<vector<int>> fourSum(vector<int>& nums, int target) {

		vector<vector<int>> vecs;
		
		if (nums.size()<4)
		{
			return vecs;
		}
		sort(nums.begin(), nums.end());
		for (int first = 0; first < nums.size() - 3;first++)
		{
			if (first>0&&nums[first]==nums[first-1])
			{
				continue;
			}

			for (int seconde = first + 1; seconde < nums.size() - 2;seconde++)
			{
				if (seconde>first+1&&nums[seconde]==nums[seconde-1])
				{
					continue;
				}

				int third = seconde + 1;
				int fouth = nums.size() - 1;

				while (third<fouth)
				{
					vector<int> temp = {nums[first],nums[seconde],nums[third],nums[fouth]};
					int sum = accumulate(temp.begin(), temp.end(), 0);
					if (sum==target)
					{
						vecs.push_back(temp);
						third++; fouth--;  //细节问题
						while (third < fouth&&nums[third] == nums[third - 1])
						{
							third++;
						}
						while (third < fouth&&nums[fouth] == nums[fouth + 1])
						{
							fouth--;
						}
	                    // third++; fouth--;放在后面有bug

						//// start++; end--放在后面就是和后面的数比较
						//while (start < end&&nums[start] == nums[start + 1])
						//{
						//	start++;
						//}
						//while (start < end&&nums[end] == nums[end - 1])
						//{
						//	end--;
						//}
						//start++; end--;

						////或者用do() while()循环
						//do{
						//	k++;
						//} while (k < l && ivec[k] == ivec[k - 1]);
						//do{
						//	l--;
						//} while (k < l && ivec[l] == ivec[l + 1]);
					}
					else if(sum<target)
					{
						third++;
					}
					else
					{
						fouth--;
					}
				}
			}
		}
		return vecs;
	}
};

// 19. Remove Nth Node From End of List
class Solution_19 {
public:
	ListNode* removeNthFromEnd(ListNode* head, int n) {

		ListNode* cur = head;
		if (head==NULL)
		{
			return NULL;
		}
		int cnt = 0;
		while (cur!=NULL)
		{
			cnt++;
			cur = cur->next;
		}
		cur = head;
		int pos = cnt - n+1;
		if (pos==1)
		{
			return cur->next;
		}
		pos = pos - 2;
		while (pos)
		{
			cur = cur->next;
			pos--;
		}
		ListNode* dete = cur->next;
		if (dete->next)
		{
			cur->next = cur->next->next;
		}
		else
		{
			cur->next = NULL;
		}
		
		
		//delete dete;
		return head;
	}
};

class Solution_20 {
public:
	bool isValid(string s) {
		if (s.size() < 1)
		{
			return true;
		}
		stack<char> sta;
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] == '(' || s[i] == '[' || s[i] == '{')
			{
				sta.push(s[i]); //入栈
			}
			else if ((s[i] == ')' || s[i] == ']' || s[i] == '}') && sta.empty())
			{
				return false;
			}
			else if ((s[i] == ')'&&sta.top() == '(') || (s[i] == ']'&&sta.top() == '[') || (s[i] == '}'&&sta.top() == '{'))
			{
				sta.pop();
			}
			else
			{
				return false;
			}
		}

		if (sta.empty())
		{
			return true;
		}
		else
		{
			return false;
		}
	}
};

class Solution_21 {
public:
	ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
		if (l1 == NULL)
		{
			return l2;
		}
		if (l2 == NULL)
		{
			return l1;
		}
		ListNode* head = new ListNode(0);
		ListNode* cur = head;
		while (l1 != NULL&&l2 != NULL)
		{
			if (l1->val <= l2->val)
			{
				cur->next = l1;
				l1 = l1->next;
				cur = cur->next;
			}
			else
			{
				cur->next = l2;
				l2 = l2->next;
				cur = cur->next;
			}
		}
		if (l1 != NULL)
		{
			cur->next = l1;
		}
		if (l2 != NULL)
		{
			cur->next = l2;
		}
		return head->next;
	}
};

class Solution_22 {
public:
	void dfs(string str, int left, int right, int total, vector<string>& vec)
	{
		if (left + right == total)
		{
			vec.push_back(str);
		}
		if (left<total / 2)  // 不能用left<=total/2等号
		{
			dfs(str + '(', left + 1, right, total, vec);
		}
		if (left>right&&right < total / 2) //左括号多余右括号
		{
			dfs(str + ')', left, right + 1, total, vec);
		}

		return;
	}

	vector<string> generateParenthesis(int n) {
		vector<string> vec;
		string str;
		if (n == 0)
		{
			return vec;
		}
		dfs("", 0, 0, 2 * n, vec);
		return vec;
	}
};

// mergeKLists
class Solution_23 {
public:
	struct CompareListNode
	{
		bool operator()(ListNode* a, ListNode* b)
		{
			return a->val > b->val;
		}
	};

	ListNode* mergeKLists(vector<ListNode*>& lists) {

		ListNode* head = new ListNode(0);
		ListNode* cur = head;
		priority_queue<ListNode*, vector<ListNode*>, CompareListNode> head_que; //元素是struct，class时，比较函数也要用对应的类型  //对于算法类型的函数调用sort();cmp使用函数即可
		for (int i = 0; i < lists.size(); i++)
		{
			if (lists[i]) //非空入堆
			{
				head_que.push(lists[i]); //建立堆
			}
		}

		while (head_que.size()>0)
		{
			ListNode* temp = head_que.top(); //取堆顶元素
			head_que.pop();
			cur->next = temp;
			cur = cur->next;

			if (temp->next)
			{
				head_que.push(temp->next);
			}
		}
		cur->next = NULL;
		return head->next;
	}
};

class Solution_24 {
public:
	/*两个指针分别指向奇数和偶数位置，一个指针指向前一次的结尾处先调换位置，再将其挂到上一次结尾处之后继续调整三个指针位置即可
	*/
	ListNode* swapPairs(ListNode* head) {
		if (head == NULL || head->next == NULL)
		{
			return head;
		}
		ListNode* newHead = NULL;
		ListNode* pre = NULL, *cur = head;
		ListNode* next = NULL;
		ListNode* temp=NULL;
		while (cur&&cur->next)
		{
			pre = cur;
			cur = cur->next;
			next = cur->next;

			pre->next = next;
			cur->next = pre;
			if (temp)
			{
				temp->next = cur; //当前挂载上一次的结尾指针上
			}	
			if (!newHead)
			{
				newHead = cur;
			}
			temp = pre; ///记录上一次的结尾指针
			cur = next;
		}

		return newHead;
	}
};

class Solution_25 {
public:
	ListNode* reverseList(ListNode* first, ListNode* tail)
	{
		ListNode* pre = NULL;
		ListNode* cur = first;
		ListNode* next = NULL;
		while (cur != tail)
		{
			next = cur->next;
			cur->next = pre; //第一个节点就为空啊！！！

			pre = cur;
			cur = next;
		}
		return first;
	}
	ListNode* reverseKGroup(ListNode* head, int k) {

		if (k < 2 || !head)
		{
			return head;
		}
		ListNode* tail = NULL;
		ListNode* first = head;

		ListNode* cur = head;
		ListNode* newHead = NULL;

		ListNode* ret = NULL;
		ListNode* pre = NULL;

		int cnt = k;
		while (cur)
		{

			while (cur&&cnt--)
			{
				/*if (cur== NULL)
				{
				break;
				}*/
				pre = cur; //
				cur = cur->next;
			}
			if (cnt > 0)
			{
				break;
			}
			if (newHead == NULL)
			{
				newHead = pre;
			}

			if (ret)
			{
				ret->next = pre; // 反转链表的尾节点直接挂下一轮k个节点的尾节点
			}
			ret = reverseList(first, cur); //cur下一次的起点，不动;返回反转后的尾节点

			ret->next = cur; //反转链表的尾节点直接挂下一轮k个节点的开始；防止下一轮没有k个节点
			first = cur;

			cnt = k;
		}

		if (!newHead)
		{
			return head;
		}

		return newHead;
	}
};

class Solution_26 {
public:

	// 此题没有说清楚，return 长度；但是nums要是去掉重复元素的
	int removeDuplicates(vector<int>& nums) {
		if (nums.size() < 2)
		{
			return nums.size();
		}
		//int cnt = 0; 
		//int j = 0;
		//for (int i = 1; i < nums.size();)
		//{
		//	if (nums[j] == nums[i])
		//	{
		//		i++;
		//	}
		//	else  //没有去掉重复元素
		//	{
		//		cnt++;
		//		j = i;
		//		i++;
		//	}
		//}
		//return cnt+1;

		int cnt = 1;
		for (int i = 1; i < nums.size(); i++)
		{
			if (nums[i] != nums[i - 1])
			{
				nums[cnt++] = nums[i];
			}
		}
		return cnt;
	}

	int removeDuplicates1(int A[], int n) {
		if (n < 2)
		{
			return n;
		}
		int cnt = 1;
		for (int i = 1; i < n; i++)
		{
			if (A[i] != A[i - 1])
			{
				A[cnt++] = A[i];
			}
		}
		return cnt;
	}
};

class Solution_27 {
public:

	// 直接查看cnt长度的元素值；若是str,最后加上‘\0’
	int removeElement(vector<int>& nums, int val) {
		int cnt = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			if (nums[i] != val)
			{
				nums[cnt++] = nums[i];
			}
		}

		return cnt;
	}


	//测试用例: 牛客网上有顺序要求
	//	[1, 2, 3, 4], 1
	//
	//	对应输出应该为 :
	//
	//			[4, 2, 3]
	//
	//你的输出为 :
	//
	//	[2, 3, 4]
	int removeElement(int A[], int n, int elem) {

		int i = 0;
		for (i = 0; i < n; i++)
		{
			if (A[i] == elem)
			{
				while (i < n&&A[--n] == elem);
				A[i] = A[n];
			}
		}

		return n;
	}
};

class Solution_28 {
public:
	// find_first_of()在源串中从位置pos起往后查找，只要在源串中遇到一个字符，该字符与目标串中任意一个字符相同，就停止查找，返回该字符在源串中的位置；若匹配失败，返回npos。
	// string查找find()函数，都有唯一的返回类型，那就是size_type，即一个无符号整数（按打印出来的算）。若查找成功，返回按查找规则找到的第一个字符或子串的位置；若查找失败，返回npos，即-1（打印出来为4294967295）
	int strStr(string haystack, string needle) { // 字符串匹配

		int ret = haystack.find(needle);

		return ret;
	}

	char *strStr1(char *haystack, char *needle) {
		int len1 = strlen(haystack);
		int len2 = strlen(needle);

		if (len1 < len2)
		{
			return NULL;
		}
		if (len2 == 0)
		{
			return haystack;
		}

		int i = 0;
		for (; i < len1 - len2 + 1; i++)
		{
			int j = 0;
			while (haystack[i+j]==needle[j])
			{
				if (j == len2 - 1)
				{
					return haystack + i;
				}
				j++;
			}
		}

		return NULL;
	}

	void getNextval(char*p,vector<int>& next)
	{
		int len = strlen(p);
		next[0] = -1;
		int k = -1; //前缀序列
		int j = 0;
		while (j < len)
		{
			if (k==-1||p[j]==p[k])
			{
				j++; k++;
				if (p[j]!=p[k])
				{
					next[j] = k;
				}
				else
				{
					next[j] = next[k];
				}
			}
			else
			{
				k = next[k];
			}
		}
	}

	char *strStr(char *haystack, char *needle)
	{
		int len1 = strlen(haystack);
		int len2 = strlen(needle);

		int i = 0, j = 0;
		vector<int> next(128,0);
		getNextval(needle, next);

		while (i<len1&&j<len2)
		{
			if (j==-1||haystack[i]==needle[j])
			{
				i++; j++;
			}
			else
			{
				j = next[j];
			}
		}
		if (j==len2)
		{
			return haystack + i-j;
		}
		return NULL;
	 }

};

// 29. Divide Two Integers
class Solution_29 { //此题保证一定除尽的条件
public:
	int divide(int dividend, int divisor) {

		if (divisor == 0 || (dividend == INT_MIN&&divisor == -1)) //考虑越界的问题
		{
			return INT_MAX;
		}
		int ret = 0;
		bool sign = (dividend > 0) ^ (divisor > 0); //异号为1

		//long long dividend_ = labs(dividend); //转换 abs(-2147483648)=-2147483648 ;换成labs在leetcode AC过但是VS2013编译器还是负数
		//long long divisor_ = labs(divisor);  //区别lab，abs,fabs()

		//long long dividend_ = llabs(long long(dividend)); //转换 abs(-2147483648)=-2147483648 //强制类型转换一个long和int 一样的，必须有两个long long

		long long dividend_ = llabs((dividend)); //用llabs()即可
		long long divisor_ = llabs(divisor);  //区别lab，abs,fabs()
		while (dividend_ >= divisor_)
		{
			int temp = divisor_;
			int multi = 1;//被除数的次数
			while (dividend_ >= (temp << 1)) //成倍的增加，时间更快
			{
				temp = temp << 1;
				multi = multi << 1;
			}
			dividend_ -= temp;
			ret += multi;
		}

		return sign ? -ret : ret;
	}
};

// 30. Substring with Concatenation of All Words
class Solution_30 {
public:
	vector<int> findSubstring(string s, vector<string>& words) {
		vector<int> vec;
		int words_num = words.size();
		int words_len = words[0].size();

		if (s.size()==0||words.size()==0||s.size()<words_num*words_len)
		{
			return vec;
		}

		unordered_map<string, int> mp;
		for (string str:words)
		{
			mp[str]++; //去掉重复元素的words
		}

		for (int i = 0; i < s.size() - words_len*words_num+1;i++)
		{
			unordered_map<string, int> dest;
			int j = 0;
			for (; j < words_num;j++) 
			{
				string temp = s.substr(i + j*words_len, words_len);
				dest[temp]++;
				if (!mp.count(temp)||(mp.count(temp)&&dest[temp]>mp[temp]))
				{
					break;
				}
			}
			if (j==words_num)
			{
				vec.push_back(i);
			}
		}
		return vec;
	}
};

// 46. Permutations 
class Solution_46 {
public:
	void help(int i,vector<int> &nums,vector<vector<int>> &vecs)
	{
		
		if (i==nums.size())
		{
			vecs.push_back(nums);
			return;
		}
		else
		{
			for (int j = i; j < nums.size();j++)
			{
				swap(nums[i],nums[j]);
				help(i + 1, nums,vecs);
				swap(nums[i],nums[j]);
			}
		}
		return;
	}

	vector<vector<int>> permute(vector<int>& nums) {

		vector<vector<int>> vecs;

		if (nums.size()==0)
		{
			return vecs;
		}

		help(0, nums,vecs);

		return vecs;
	}
};

// Permutations ii
class Solution_47 {
public:

	bool IsSwap(vector<int>&nums,int i,int j)
	{
		for (; i < j; i++)
		{
			if (nums[i] == nums[j])
			{
				return false;
			}
		}
		return true;
	}

	void help(int i,vector<int>&nums,vector<vector<int>> &vecs)
	{
		if (i==nums.size())
		{
			vecs.push_back(nums);
			return;
		}
		for (int j = i; j < nums.size(); j++)
		{
			if (IsSwap(nums,i,j))
			{
				swap(nums[i],nums[j]);
				help(i + 1, nums, vecs);
				swap(nums[i],nums[j]);
			}
		}
		return;
	}

	vector<vector<int>> permuteUnique(vector<int>& nums) {
		vector<vector<int>> vecs;
		vector<int> vec;

		if (nums.size()==0)
		{
			return vecs;
		}

		help(0,nums,vecs);

		return vecs;
	}
};

// 31. Next Permutation
class Solution_31 {
public:
	void nextPermutation(vector<int> &num) {

		next_permutation(num.begin(), num.end());

		return;
	}
};

// longestValidParentheses(括号)
class Solution_32 {
public:
	// 本题使用初始-1入栈，就省去了记录记录当前括号匹配的子串的左侧位置;（（））：3-(-1)=4; 但是栈为空的时候，需要入栈当前位置，例如 ())()()=6-2=4，在str[2]='）'入栈index
	// 只是入栈‘（’并不能解决问题，需要入栈‘（’的下标index
	int longestValidParentheses(string s) {
		if (s.size()<=1)
		{
			return 0;
		}
		stack<int> sta;
		sta.push(-1); //技巧使用了初始化，否则需要记录 需要借助变量l记录当前括号匹配的子串的左侧位置
		int ret = 0;
		for (int i = 0; i < s.size();i++)
		{
			if (s[i]=='(')
			{
				sta.push(i);
			}
			else
			{
				sta.pop();
				if (sta.empty())
				{
					sta.push(i); //入栈
				}
				else
				{
					int cur = i - sta.top();
					ret = max(ret,cur);
				}
			}
			
		}
		return ret;
	}
};

// 33. Search in Rotated Sorted Array
class Solution_33 {
public:
	// 本质:不管什么情况，都是只是low,high进行移动，二分查找时候一定记住要有常数步的前进，防止进入死循环

//input : [3, 1]
//       	1
//Output : -1 bug1:加等号
//	 Expected : 1
	int search(vector<int>& nums, int target) {
		if (nums.size()==1&&nums[0]==target)
		{
			return 0;
		}
		int low = 0, high = nums.size()-1;
		while (low<=high)
		{
			int mid = low + (high - low) / 2;
			if (nums[mid]==target)
			{
				return mid;
			}
			if (nums[mid]>target)
			{
				if (nums[low]<=nums[mid]) //低半部分有序;  bug 1有序部分要用等号
				{
					if (nums[low]<=target) //target在低半部分序列中
					{
						high = mid - 1;
					}
					else
					{
						low = mid + 1;
					}
				}
				else // 后半部分有序，且nums[mid]>target;必位于前半部分
				{
					high = mid - 1;
				}
			}
			else
			{
				if (nums[mid]<=nums[high]) //后半部分有序
				{
					if (nums[high]>=target)
					{
						low = mid + 1;
					}
					else
					{
						high = mid - 1;
					}
				}
				else //nums[mid]>target 且前部分有序
				{
					low = mid + 1;
				}
			}
			
		}
		return -1;
	}

	int search2(int A[], int n, int target) {

		int low = 0, high = n - 1;

		while (low<=high)
		{
			int mid = low + (high - low) >> 1;
			if (A[mid]==target)
			{
				return mid;
			}
			if (A[mid]>=A[low])  //低半部分有序；先比较区间，在比较关键字target
			{
				if (A[mid]>target&& target>=A[low]) //bug 2: 调整那个，就不用等号
				{
					high = mid - 1;
				}
				else
				{
					low = mid + 1;
				}
			}
			else //后半部分有序
			{
				if (A[mid]<target&&target<=A[high])
				{
					low = mid + 1;
				}
				else
				{
					high = mid - 1;
				}
			}
		}
		return -1;
	}

	int search_ref(vector<int>& nums, int target) {
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

// searchRange
class Solution_34 {
public:
	int findUpBound(vector<int>& nums, int target) //找上边界
	{
		int low = 0, high = nums.size() - 1;
		if (nums.back()<target)
		{
			return -1;
		}
		while (low<=high)
		{
			int mid = (low + high) >> 1;
			if (nums[mid]<=target)   /// 向右夹逼，找到右边界
			{
				low = mid + 1;
			}
			else
			{
				high = mid - 1;
			}
		}
		return high; //向右夹逼，返回high
	}

	int findDownBound(vector<int>& nums, int target) // 找下边界
	{
		int low = 0, high = nums.size() - 1;
		if (nums.front()>target)
		{
			return -1;
		}
		while (low<=high)
		{
			int mid = (low + high) >> 1;
			if (nums[mid]<target)   /// 向左夹逼，则找到左边界
			{
				low = mid + 1;
			}
			else
			{
				high = mid - 1;
			}
		}

		return low; // 向左夹逼，返回low
	}

	vector<int> searchRange(vector<int>& nums, int target) { //找到target的上下边界
		vector<int> vec(2, -1);
		if (nums.size() == 0 || nums.front()>target || nums.back()<target)
		{
			return vec;
		}

		int high = findUpBound(nums, target);
		int low = findDownBound(nums, target);
		
		if ((low == high&&nums[low] == target) || (low<high))
		{
			vec[0] = low;
			vec[1] = high;
		}
		
		return vec;
	}
  

	vector<int> searchRange2(int A[], int n, int target) {
		vector<int > vec(A,A+n); //初始化

		return searchRange1(vec, target);

	}

	vector<int> searchRange1(vector<int>& nums, int target) { //找到target的自身上下边界，非严格上下界
		vector<int> vec(2, -1);
		if (nums.size() == 0||nums.front()>target||nums.back()<target)
		{
			return vec;
		}

		//pair<int, int> temp;
		//针对 upper_bound the objects in the range [first,last) are accessed. 本身传入的迭代器end().就是指向下一个位置
		auto range=equal_range(nums.begin(), nums.end(), target); //返回的是两个迭代器,解引用得到值
		
		if (*range.first!=target||*(range.second-1)!=target)
		{
			return vec; 
		}
		vec[0]=(range.first-nums.begin()); //stl: distance()计算迭代器之间的距离
		vec[1]=(range.second - nums.begin()-1);

		return vec;
	}
};

// 35. Search Insert Position
class Solution_35 {
public:
	int searchInsert1(vector<int>& nums, int target) {

		int low = 0, high = nums.size() - 1;
		while (low<=high)
		{
			int mid = (low + high) >> 1;
			if (nums[mid]==target)
			{
				return mid;
			}
			else if (nums[mid]>target)
			{
				high = mid - 1;
			}
			else
			{
				low = mid + 1;
			}
		}
		return low; //没有查找到，返回的low,即插入位置
	}

	int searchInsert(int A[], int n, int target) {
		vector<int> vec(A, A + n);
		return searchInsert1(vec, target);
	}
};

// 36. Valid Sudoku 数独/九宫格问题
class Solution_36 {
public:
	bool isValidSudoku(vector<vector<char>>& board) {

		int row = board.size();
		int col = board[0].size();

		unordered_map<char,int> row_mp; // <char, bool>
		unordered_map<char,int> col_mp;
		unordered_map<char, int>  diagonal; //对角线; sub-boxes

		for (int i = 0; i < row;i++)
		{
			for (int j = 0; j < col;j++)
			{
				if (board[i][j]!='.' && row_mp.find(board[i][j])!=row_mp.end())
				{
					return false; //已经存在
				}
				else
				{
					row_mp[board[i][j]]++;
				}
				if (board[j][i] != '.'&& col_mp.find(board[j][i]) != col_mp.end())
				{
					return false;
				}
				else
				{
					col_mp[board[j][i]]++;
				}
				if (board[i / 3 * 3 + j / 3][i % 3 * 3 + j % 3] != '.'&& diagonal.find(board[i / 3 * 3 + j / 3][i % 3 * 3 + j % 3]) != diagonal.end()) // //第i个九宫格第j个格子
				{
					return false;
				}
				else
				{
					diagonal[board[i / 3 * 3 + j / 3][i % 3 * 3 + j % 3]]++;
				}
			}
			col_mp.clear();
			row_mp.clear();
			diagonal.clear();
		}
		return true;
	}
};

// 37. Sudoku Solver
class Solution_37 {
public:
	bool isValid(vector<vector<char>>& board,int i,int j) //只需要判断当前行，列，方格是否合法，节省时间
	{
		for (int row = 0; row < 9;row++)
		{
			if (row!=i&&board[i][j]==board[row][j])
			{
				return false;
			}
		}
		for (int col = 0; col < 9;col++)
		{
			if (col!=j&&board[i][j]==board[i][col])
			{
				return false;
			}
		}
		for (int row = i / 3 * 3; row < i / 3 * 3 + 3;row++)
		{
			for (int col = j / 3 * 3; col < j / 3 * 3 + 3;col++)
			{
				if ((row!=i||col!=j)&&board[i][j]==board[row][col]) //小方格也需要行列判断
				{
					return false;
				}
			}
		}
		return true;
	}

	bool dfs(vector<vector<char> > &board,int i,int j)
	{
		if (i==9)
		{
			return true;
		}
		if (j==9)
		{
			return dfs(board, i + 1, 0);
		}

		if (board[i][j]=='.')
		{
			for (char k = '1'; k <='9';k++)
			{
				board[i][j] = k;
				if (isValid(board,i,j))
				{
					if (dfs(board, i, j + 1))
					{
						return true;
					}
				}
				board[i][j] = '.'; //达到回溯的目的
			}
		}
		else
		{
			return dfs(board, i, j + 1);
		}
		return false;
	}

	void solveSudoku(vector<vector<char>>& board) {

		if (board.size()<9||board[0].size()<9)
		{
			return;
		}
		dfs(board, 0, 0);
		return;
	}
};

// add 39. Combination Sum
class Solution_39 {
public:

	void dfs(vector<vector<int>> &vecs, vector<int> &vec, int i, int target, vector<int> &candidates)
	{
		if (target==0)
		{
			vecs.push_back(vec);
			return;
		}
		if (target<0)
		{
			return;
		}
		for (int k = i; k < candidates.size(); k++)
		{
			vec.push_back(candidates[k]);
			dfs(vecs, vec, k, target - candidates[k], candidates);
			vec.pop_back();
		}
		return;
	}

	// 默认有序
	vector<vector<int> > combinationSum(vector<int> &candidates, int target) {
		vector<vector<int>> vecs;
		vector<int> vec;
		if (candidates.size()==0)
		{
			return vecs;
		}
		sort(candidates.begin(), candidates.end());
		dfs(vecs, vec, 0, target, candidates);

		return vecs;
 	}
};

// add 40. Combination Sum ii
class Solution_40 {
public:

	void dfs(vector<vector<int>> &vecs, vector<int> &vec, int i, int target, vector<int> &candidates)
	{
		if (target == 0)
		{
			vecs.push_back(vec);
			return;
		}
		if (target < 0)
		{
			return;
		}
		// 1 1 2 5 6 (1,1,6;1 2 5)
		for (int k = i; k < candidates.size(); k++)
		{
			if (k>i && candidates[k] == candidates[k - 1]) //k>0 bug
			{
				continue;
			}
			vec.push_back(candidates[k]);
			dfs(vecs, vec, k+1, target - candidates[k], candidates);
			vec.pop_back();
		}
		return;
	}

	// 默认有序
	vector<vector<int> > combinationSum2(vector<int> &candidates, int target) {
		vector<vector<int>> vecs;
		vector<int> vec;
		if (candidates.size() == 0)
		{
			return vecs;
		}
		sort(candidates.begin(), candidates.end());
		dfs(vecs, vec, 0, target, candidates);

		return vecs;
	}
};

// add 38. Count and Say
class Solution_38 {
public:
	string countAndSay(int n) {
		if (n<=0)
		{
			return "";
		}
		string res="1";
		for (int i = 1; i < n;i++) //依次迭代n-1次
		{
			string temp = "";
			int cnt = 1;
			for (int j = 1; j < res.length();j++)
			{
				if (res[j-1]==res[j])
				{
					cnt++;
				}
				else
				{
					temp.push_back(cnt + '0'); //相同元素的个数
					temp.push_back(res[j-1]); //相同的元素是什么！
					cnt = 1; //元素个数复位
				}
			}
			temp.push_back(cnt + '0');
			temp.push_back(res[res.length()-1]); //最后一个元素
			res = temp;//重新赋值下一轮迭代值
		}

		return res;
	}
};

// add 41. First Missing Positive
class Solution_41 {
public:
	//对于每个数i调整到i-1位置，当然调整后还要接着判断。
	//最后重头扫一遍，发现第一个a[i]!=i+1位置的就返回i+1；

	int firstMissingPositive(vector<int>& nums) { //数组约定为1-n吗？
		
		for (int i = 0; i < nums.size();i++)
		{
			if (nums[i]>0&&nums[i]<=nums.size()&&nums[i]!=nums[nums[i]-1]) // [3,4,-1,1] //用if不能一直将元素放入正确的位置，eg：1需要交换两次才行
			while (nums[i]>0 && nums[i] <= nums.size() && nums[i] != nums[nums[i] - 1]) 
			{
				swap(nums[i],nums[nums[i]-1]);
			}
		}

		for (int i = 0; i < nums.size();i++)
		{
			if (nums[i]!=i+1)
			{
				return i+1;
			}
		}
		return nums.size()+1; // 空或者本身有序，缺失最后一个
	}

	int firstMissingPositive(int A[], int n) {
		for (int i = 0; i < n; i++)
		{
			while (A[i] > 0 && A[i] <= n && A[i] != A[A[i] - 1])
			{
				swap(A[i], A[A[i] - 1]);
			}
		}

		for (int i = 0; i < n; i++)
		{
			if (A[i] != i + 1)
			{
				return i + 1;
			}
		}
		return n + 1; // 空或者本身有序，缺失最后一个
	}
};

// add 42. Trapping Rain Water
class Solution_42 {
public:

	// 本来自己想用总面积-黑色块的面积，但是总面积不容易求得
	// 思路1：找到最高的柱子，分左右两边处理 
	int trap(vector<int>& height) {

		if (height.size()<=0)
		{
			return 0;
		}
		int max_index = 0;
		int max_height = height[0];
		for (int i = 1; i < height.size();i++)
		{
			if (height[i]>max_height)
			{
				max_height = height[i];
				max_index = i;
			}
		}

		int sum = 0;
		int max_left = 0;
		for (int i = 0; i < max_index;i++)
		{
			if (height[i]>max_left)
			{
				max_left = height[i];
			}
			else
			{
				sum += (max_left-height[i]);
			}
		}

		int max_right = 0;
		for (int i = height.size() - 1; i >max_index; i--)
		{
			if (height[i]>max_right)
			{
				max_right = height[i];
			}
			else
			{
				sum += (max_right-height[i]);
			}
		}
		return sum;
	}

	int trap(int A[], int n) {
		vector<int > vec(A,A+n);
		
		return trap(vec);

	}
};

#include <string>
// add 43. Multiply Strings
class Solution_43 {
public:
	string multiply(string num1, string num2) {
		int len1 = num1.size();
		int len2 = num2.size();

		vector<int> vec(len1+len2,0); // 初始化内存空间
		//vec.reserve(len1 + len2);
		
		for (int i = 0; i < len1; i++)
		{
			int k = i;
			for (int j = 0; j < len2;j++)
			{
				//vec.push_back(a*b);
				vec[k] += (num1[len1-1-i] - '0')*(num2[len2-1-j]-'0');	////Calculate from rightmost to left
				k++; 
			}
		}

		string ret="";
		for (int i = 0; i < vec.size();i++)
		{
			if (vec[i]>=10)
			{
				vec[i+1] += vec[i] / 10;
				vec[i] = vec[i] % 10;
			}
			//char temp[5];
			//_itoa(vec[i], temp, 10);
			//ret += temp;

			char temp = vec[i] + '0'; 
			ret += temp; //反着取得
		}

		//reserve(ret.begin(),ret.end());
		//reverse(vec.begin(), vec.end()); //string没有反转函数，vector有
		
		//判断第一个非0位 //size_t startpos = sum.find_first_not_of("0");
		int flag = 0;
		for (int i = ret.size()-1; i >=0;i--)
		{
			if (ret[i]!='0')
			{
				flag = i;
				break;
			}
		}

		int begin = 0, end = flag;
		while (begin < end)
		{
			swap(ret[begin], ret[end]);
			begin++;
			end--;
		}

		return ret.substr(0,flag+1);
	}
};

// add 44. Wildcard Matching
class Solution_44 {
public:
	bool isMatch(string s, string p) {
		int i = 0, j = 0, j_recall = 0, i_recall = 0;
		while (s[i]) {
			if (p[j] == '?' || s[i] == p[j])
			{
				++i; ++j; continue;
			}
			if (p[j] == '*')
			{
				i_recall = i; j_recall = ++j; continue;
			}
			if (j_recall)
			{
				i = ++i_recall; j = j_recall; continue;
			}
			return
				false;
		}
		while (p[j] == '*')
			++j;
		return !p[j];
	}
};

// 45. Jump Game II
class Solution_45 {
public:
	int jump(vector<int>& nums) {
		int n = nums.size(), step = 0;
		int start = 0, end = 0; //bfs每一层的开始结束位置 //每层结束更新
		while (end<n-1) //end<n时，end=n-1就可以结束了
		{
			step++;
			int maxend = end + 1;
			for (int i = start; i <= end;i++)
			{
				if (i+nums[i]>n)
				{
					return step;
				}
				maxend = max(i+nums[i],maxend);
			}
			start = end + 1;
			end = maxend;
		}
		return step;
	}

	int jump(int A[], int n) {

		vector<int> vec(A,A+n);
		return jump(vec);
	}
};

class Solution_48 {
public:
	void rotate(vector<vector<int>>& matrix) {

		// 总的位置坐标关系：rotate[j][n-1-i]=a[i][j]
		int n = matrix.size();
		reverse(matrix.begin(), matrix.end()); //a[n-1-i][j]=a[i][j] 
		for (int i = 0; i < n;i++)
		{
			for (int j = i + 1; j < n;j++)
			{
				swap(matrix[i][j],matrix[j][i]); // a[i][j]=a[j][i]
			}
		}
		return;
	}
};

// add 49. Group Anagrams
class Solution_49 {
public:
	// 回文构词法有一个特点：单词里的字母的种类和数目没有改变，只是改变了字母的排列顺序。

	vector<vector<string> > groupAnagrams(vector<string> &strs) {
		vector<vector<string> > vecs;
		vector<string> vec;

		int len = strs.size();
		unordered_map<string, vector<string>> mp;
		for (int i = 0; i < len;i++)
		{
			string temp = strs[i];
			sort(temp.begin(), temp.end());
			mp[temp].push_back(strs[i]);
		}

		for (auto &iter:mp)
		{
			vecs.push_back(iter.second);
		}
		return vecs;
	}

	vector<string> anagrams(vector<string> &strs) {

		vector<string> vec;

		int len = strs.size();
		if (len<=0)
		{
			return vec;
		}
		unordered_map<string, vector<string>> mp;
		for (int i = 0; i < len; i++)
		{
			string temp = strs[i];
			sort(temp.begin(), temp.end());
			mp[temp].push_back(strs[i]);
		}

		for (auto iter = mp.begin(); iter != mp.end();iter++)
		{
			if (iter->second.size()>1)
			    vec.insert(vec.end(), iter->second.begin(), iter->second.end());
		}
		return vec;
	}
};

// add 50. Pow(x, n)
class Solution_50 {
public:

	double myPow(double x, int n) { //n正负之分

		double ret = 1.0;
		if (n == 0)
		{
			return (double)1;
		}
		else if (n > 0)
		{
			while (n--) //超时
			{
				ret *= x;
			}
		}
		else
		{
			x = 1 / x;
			while (n++)
			{
				ret *= x;
			}
		}

		return ret;
	}
	double myPow2(double x, int n) { //n正负之分

		double ret = 1.0;
		if (n==0)
		{
			return (double)1;
		}
		if (n<0)
		{
			return 1 / x*myPow(x, n + 1); //递归太深
		}
		else
		{
			return x*myPow(x, n - 1);
		}

		return ret;
	}

	double myPow1(double x, int n) {
		double ret = 1.0;
		if (n == 0)
		{
			return (double)1;
		}
		if (n<0) //负数处理
		{
			ret = 1 / x; //防止最MIN溢出
			x = 1 / x;
			n = -n - 1;
		}

		if (n%2==0)
		{
			;
		}
		else
		{
			ret *= x; 
			n--;
		}
		double temp = myPow(x, n / 2);
		ret *= (temp*temp);

		return ret;
	}

	// 链接：https://www.nowcoder.com/questionTerminal/0616061711c944d7bd318fb7eaeda8f6
	double pow(double x, int n) {
		if (n == 0) return 1;
		if (n < 0) 
			return 1 / x * pow(1 / x, -(n + 1));
		if (n % 2 == 0) 
			return pow(x * x, n / 2);
		else 
			return pow(x * x, n / 2) * x;
	}
};

// add 51. N-Queens
class Solution_51 {
public:
	bool isValid(vector<string> &vec,int i,int j)
	{
		//判断当前放置位置（i，j）是否合理
		for (int row = 0; row < i;row++) //判断同一列列
		{
			if (vec[row][j]=='Q') // vec[i][j]
			{
				return false;
			}
		}
		//判断对角线45°
		for (int row = i-1, col = j-1; row >= 0 && col>=0;row--,col--)
		{
			if (vec[row][col] == 'Q')
			{
				return false;
			}
		}
		//判断对角线135%
		for (int row = i-1, col = j + 1; row >= 0 && col < vec.size();row--,col++) //写法： row >= 0, col < vec.size() 错误
		{
			if (vec[row][col] == 'Q')
			{
				return false;
			}
		}
		return true;
	}

	void solveNQueensHelp(vector<vector<string>> &vecs,vector<string> &vec,int row,int n)
	{
		if (row==n)
		{
			vecs.push_back(vec);
			return;
		}

		for (int col = 0; col < n;col++)
		{
			if (isValid(vec,row,col))
			{
				vec[row][col] = 'Q';
				solveNQueensHelp(vecs, vec, row + 1, n);
				vec[row][col] = '.';
			}
		}
		return;
	}

	vector<vector<string>> solveNQueens(int n) {

		vector<string> vec(n,string(n,'.')); 

		vector<vector<string> > vecs; //所有解

		solveNQueensHelp(vecs,vec,0,n);
		
		return vecs;
	}
};

class Solution_52 {
public:
	int ret = 0;

	bool isValid(vector<string> &vec, int i, int j)
	{
		//判断当前放置位置（i，j）是否合理
		for (int row = 0; row < i; row++) //判断同一列列
		{
			if (vec[row][j] == 'Q') // vec[i][j]
			{
				return false;
			}
		}
		//判断对角线45°
		for (int row = i - 1, col = j - 1; row >= 0 && col >= 0; row--, col--)
		{
			if (vec[row][col] == 'Q')
			{
				return false;
			}
		}
		//判断对角线135%
		for (int row = i - 1, col = j + 1; row >= 0 && col < vec.size(); row--, col++) //写法： row >= 0, col < vec.size() 错误
		{
			if (vec[row][col] == 'Q')
			{
				return false;
			}
		}
		return true;
	}

	void solveNQueensHelp( vector<string> &vec, int row, int n)
	{
		if (row == n)
		{
			ret++;
			return;
		}

		for (int col = 0; col < n; col++)
		{
			if (isValid(vec, row, col))
			{
				vec[row][col] = 'Q';
				solveNQueensHelp( vec, row + 1, n);
				vec[row][col] = '.';
			}
		}
		return;
	}

	
	int totalNQueens(int n) {

		vector<string> vec(n, string(n, '.'));

		solveNQueensHelp(vec, 0, n);

		return ret;
	}


	// 链接：https://www.nowcoder.com/questionTerminal/00b9b6bb397949b0a56d2bc351c4cf23
	bool isValid(vector<int> &pos, int row, int col)
	{
		for (int i = 0; i < row; ++i) {
			if (col == pos[i] || abs(row - i) == abs(col - pos[i])) { //在同一列，或者行相减=列相减
				return false;
			}
		}
		return true;
	}

	void solveNQueensDFS(vector<int> &pos, int row, int &res)
	{
		int n = pos.size();
		if (row == n) {
			res++;
		}
		else {
			for (int col = 0; col < n; ++col) {
				if (isValid(pos, row, col)) {
					pos[row] = col;
					solveNQueensDFS(pos, row + 1, res);
					pos[row] = -1;
				}
			}
		}
	}

	int totalNQueens2(int n)
	{
		int res = 0;
		vector<int> pos(n, -1);//记录每一行的Queen所在的位置（0-row）
		solveNQueensDFS(pos, 0, res);
		return res;
	}


};

// add 53. Maximum Subarray
class Solution {
public:
	int maxSubArray(vector<int>& nums) {

		int ret = INT_MIN;
		int temp = 0;
		for (int i = 0; i < nums.size();i++)
		{
			temp += nums[i];

			if (temp>ret) //两个if的先后顺序
			{
				ret = temp;
			}

			if (temp < 0)
			{
				temp = 0;
			}
		}
		return ret;
	}

	int maxSubArray(int A[], int n) {
		vector<int> vec(A,A+n);
		return maxSubArray(vec);
	}
};

class Solution_54 {
public:
	void help(vector<vector<int>>& matrix,vector<int> &res, int x0, int y0, int x1, int y1)
	{
		if (x0==x1&&y0==y1)
		{
			res.push_back(matrix[x0][y0]);
		}else if (x0==x1) //最后一行
		{
			for (int i = y0; i <= y1;i++)
			{
				res.push_back(matrix[x0][i]);
			}
		}else if (y1==y0)
		{
			for (int i = x0; i <= x1;i++)
			{
				res.push_back(matrix[i][y0]);
			}
		}
		else
		{
			for (int col = y0; col <= y1;col++) 
			{
				res.push_back(matrix[x0][col]);
			}
			for (int row = x0 + 1; row <= x1;row++)
			{
				res.push_back(matrix[row][y1]);
			}
			for (int col = y1 - 1; col >= y0;col--)
			{
				res.push_back(matrix[x1][col]);
			}
			for (int row = x1 - 1; row > x0;row--)
			{
				res.push_back(matrix[row][y0]);
			}
		}
		return;
	}

	vector<int> spiralOrder(vector<vector<int>>& matrix) {
		vector<int> res;
		if ( matrix.empty()) //
		{
			return vector<int>(); //res;
		}
		int row = matrix.size();
		int col = matrix[0].size(); //后面判断错误：row <= 0 || col <= 0 ，需要计算row,col调用size()出错，需要在函数开始判断

		int x0 = 0, y0 = 0, x1 = row - 1, y1 = col - 1;
		while (x0<=x1&&y0<=y1)
		{
			help(matrix,res, x0, y0, x1, y1);
			x0++;
			y0++;
			x1--;
			y1--;
		}

		return res;
	}
};

class Solution_55 {
public:
	bool canJump(vector<int>& nums) {

		int maxend = 0;
		for (int i = 0; i < nums.size()&&maxend>=i;i++)
		{
			maxend = max(maxend,i+nums[i]);
		}

		if (maxend<nums.size()-1)
		{
			return false;
		}
		return true;
	}

	bool canJump(int A[], int n) {
		vector<int> vec(A, A + n);
		return canJump(vec);
	}
};

//Definition for an interval.
struct Interval {
	int start;
	int end;
	Interval() : start(0), end(0) {}
	Interval(int s, int e) : start(s), end(e) {}
};

class Solution_56 {
public:

	static int compare(Interval val1,Interval val2)
	{
		return val1.start < val2.start;
	}

	vector<Interval> merge(vector<Interval>& intervals) {

		if (intervals.size()<=1)
		{
			return intervals;
		}
		sort(intervals.begin(),intervals.end(),compare); //按第一关键字排序
		vector<Interval> vec;
		Interval temp=intervals[0];
		
		for (int i = 1; i < intervals.size(); i++)
		{
			Interval node = intervals[i]; //取出每一个节点
			if (node.start<=temp.end)
			{
				temp.end = max(temp.end,node.end);  // [[1,4],[2,3]]
			}
			else
			{
				vec.push_back(temp);
				temp = intervals[i];
			}
		}
		vec.push_back(temp);
		return vec;
	}
};

class Solution_57 {
public:
	static int compare(Interval val1, Interval val2)
	{
		return val1.start < val2.start;
	}

	vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {

		vector<Interval> vec;
		if (intervals.empty())
		{
			vec.push_back(newInterval);
			return vec;
		}
		intervals.push_back(newInterval);
		sort(intervals.begin(), intervals.end(), compare);

		Interval node = intervals[0]; //
		for (int i = 1; i < intervals.size();i++)
		{
			Interval temp = intervals[i];
			if (node.end>=temp.start)
			{
				node.end = max(node.end,temp.end);
			}
			else
			{
				vec.push_back(node);
				node = temp;
			}
		}
		vec.push_back(node);

		return vec;
	}
};

class Solution_58 {
public:

	// 反向查找，末尾空格忽略，行中出现空格就终止循环

	int lengthOfLastWord(string s) {
		int ret = 0;
		if (s.empty())
		{
			return ret;
		}
		int i = s.size() - 1;
		while (i>=0&&s[i] == ' ')
		{
			i--;
		}
			
		for (; i >= 0;i--)
		{	
			if (s[i]==' ')
			{
				break;
			}
			ret++;
		}
		return ret;
	}

	int lengthOfLastWord(const char *s) {

		int ret = 0;
		int len = strlen(s);

		for (int i = len - 1; i >= 0;i--)
		{
			if (s[i]==' ')
			{
				if (ret) //忽略末尾的空格，当遇到空格且有元素时，返回
				{
					break;
				}
			}
			else
			{
				ret++;
			}
		}
		return ret;
	}
};

// 59. Spiral Matrix II
class Solution_59 {
public:
	vector<vector<int> > generateMatrix(int n) {

		vector<vector<int>> vecs(n, vector<int>(n,1));
		
		int x0 = 0, y0 = 0;
		int x1 = n - 1, y1 = n - 1;

		int index = 0;
		while (x0<=x1&&y0<=y1)
		{
			for (int i = y0; i <= y1;i++)
			{
				vecs[x0][i] = ++index;
			}
			for (int i = x0 + 1; i <= x1;i++)
			{
				vecs[i][y1] = ++index;
			}
			for (int i = y1 - 1; i >= y0;i--)
			{
				vecs[x1][i] = ++index;
			}
			for (int i = x1 - 1; i > x0;i--)
			{
				vecs[i][y0] = ++index;
			}

			x0++, y0++;
			x1--, y1--;
		}
		
		return vecs;
	}
};

// 60. Permutation Sequence
class Solution_60 {
public:
	string getPermutation(int n, int k) {

		vector<int> vec;
		for (int i = 0; i < n;i++)
		{
			vec.push_back(i + 1);
		}
		
		for (int i = 0; i < k-1;i++)
		{
			next_permutation(vec.begin(), vec.end());
		}

		string res;
		for (int i = 0; i < vec.size();i++)
		{
			char temp = vec[i] + '0';
			res.push_back(temp);
		}
		
		return res;
	}

	string getPermutation_ref(int n, int k) {
		vector<int> permutation(n + 1, 1);
		for (int i = 1; i <= n; ++i) {
			permutation[i] = permutation[i - 1] * i;
		}
		vector<char> digits = { '1', '2', '3', '4', '5', '6', '7', '8', '9' };
		int num = n - 1;
		string res;
		while (num) {
			int t = (k - 1) / (permutation[num--]);
			k = k - t * permutation[num + 1];
			res.push_back(digits[t]);
			digits.erase(digits.begin() + t);
		}
		res.push_back(digits[k - 1]);
		return res;
	}

};

class Solution_61 {
public:
	//input:[1, 2]  3
	//output:[2, 1]
	ListNode *rotateRight(ListNode *head, int k) {

		if (!head||k==0||!head->next)
		{
			return head;
		}

		ListNode* newHead = head;
		ListNode* fast = head;
		ListNode* lastNode = NULL;
		int len = 0;
		while (fast!=nullptr)
		{
			if (fast->next==NULL)
			{
				lastNode = fast;
			}
			fast = fast->next;
			len++;
		}

		fast = head;

		int step = len - k%len-1; // 注意k可能会大于len，因此k%=len
		if (k%len == 0)
		{
			return newHead;
		}
		while (step)
		{
			step--;
			fast = fast->next;
		}

		newHead = fast->next;
		fast->next = NULL;
		lastNode->next = head;

		return newHead;
	}
};

class Solution_62 {
public:
	int uniquePaths(int m, int n) {
		//matrix(m*n)
		vector<vector<int>> vecs(m, vector<int>(n, 1));

		for (int i = 1; i < m;i++)
		{
			for (int j = 1; j < n;j++)
			{
				vecs[i][j] = vecs[i - 1][j] + vecs[i][j - 1];
			}
		}
		return vecs[m-1][n-1];
	}

	int uniquePaths1(int m, int n) {
		vector<int > vec(n, 1); //压缩空间
		for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		if (i * j != 0)
			vec[j] += vec[j - 1];
		return vec[n - 1];
	}

//      链接：https://www.nowcoder.com/questionTerminal/166eaff8439d4cd898e3ba933fbc6358
//		动态规划的复杂度也是n方，可以用排列组合的方式，复杂度为n
//		只能向右走或者向下走，所以从一共需要的步数中挑出n - 1个向下走，剩下的m - 1个就是向右走
//		其实就是从（m - 1 + n - 1）里挑选（n - 1）或者（m - 1）个，c(n, r)     n = （m - 1 + n - 1）, r = （n - 1）
//		n!/ (r!* (n - r)!)

	//注意观察到，可以发现循环的值是；C(n, m) = n!/ (m!*(n - m)!)，因为n值过大，不可以直接用公式
    //组合数学的递推公式：C(m,n)=C(m,n-1)+C(m-1,n-1)
	//C(n, 1) = n; C(n, n) = 1; C(n, 0) = 1;这样就可以用DP了

	int fun(int n, int m)
	{
		if (m==1)
		{
			return n;
		}
		if (n==m||m==0)
		{
			return 1;
		}
		return fun(n-1, m ) + fun(n - 1, m - 1);	//超时
	}
	int uniquePaths2(int m, int n) {
		
		n = (m - 1 + n - 1);
		m = (m - 1);
		
		int ret=fun(n,m);

		return ret;
	}


};

class Solution_63 {
public:
	int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid) {

		/// 使用O(n)空间的方案
		int m = obstacleGrid.size(), n = obstacleGrid[0].size();
		if (m == 0 || n == 0)
			return 0;
		vector<int> res(n, 0);
		res[0] = 1;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j<n; j++)
			{
				if (obstacleGrid[i][j] == 1)
					res[j] = 0;
				else if (j>0)
					res[j] = res[j] + res[j - 1];
			}
		}
		return res[n - 1];

	}
};


//动态规划法求最长递增子序列 LIS  
int dp[101]; /* 设数组长度不超过100，dp[i]记录到[0,i]数组的LIS */
int lis;    /* LIS 长度 */
int LIS(int * arr, int size)
{
	for (int i = 0; i < size; ++i)
	{
		dp[i] = 1;
		for (int j = 0; j < i; ++j)
		{
			if (arr[i] > arr[j] && dp[i] < dp[j] + 1)
			{
				dp[i] = dp[j] + 1;
				if (dp[i] > lis)
				{
					lis = dp[i];
				}
			}
		}
	}
	return lis;
}

// add 64. Minimum Path Sum
class Solution_64 {
public:
	int minPathSum(vector<vector<int>>& grid) {

		int m = grid.size();
		int n = grid[0].size();

		vector<vector<int>> dp(m, vector<int>(n, 0));
		
		for (int i = 0; i < m;i++)
		{
			for (int j = 0; j < n;j++)
			{
				if (i==0&&j==0)
				{
					dp[i][j] = grid[0][0];
				}else if (i==0)
				{
					dp[i][j] = dp[i][j - 1]+grid[i][j];
				}
				else if (j==0)
				{
					dp[i][j] = dp[i-1][j] + grid[i][j];
				}
				else
				{
					dp[i][j] = min(dp[i-1][j], dp[i][j - 1]) + grid[i][j];
				}
				
			}
		}
		return dp[m-1][n-1];
	}
};


class Solution_21_ {
public:
	ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
		if (l1==NULL)
		{
			return l2;
		}
		if (l2==0)
		{
			return l1;
		}

		ListNode* newHead = new ListNode(0);
		ListNode* cur = newHead;
		while (l1!=NULL&&l2!=NULL)
		{
			if (l1->val>=l2->val)
			{
				cur->next = l2;
				l2 = l2->next;
			}
			else
			{
				cur->next = l1;
				l1 = l1->next;
			}
			cur = cur->next;
		}
		if (l1!=NULL)
		{
			cur->next = l1;
		}
		if (l2!=NULL)
		{
			cur->next = l2;
		}
		
		return newHead->next;

	}

};

// 67. Add Binary
class Solution_67 {
public:
	string addBinary(string a, string b) {

		string ret;
		int len = (a.size() < b.size()) ? a.size() : b.size();
		bool falg = false; //进位标志
		reverse(a.begin(), a.end());
		reverse(b.begin(), b.end());

		for (int i = 0; i < len;i++)
		{
			if (a[i]=='1'&&b[i]=='1')
			{
				if (falg)
				{
					ret.push_back('1');
				}
				else
				{
					ret.push_back('0');
				}
				falg = true;
			}
			else if ((a[i] == '1'&&b[i] == '0') || (a[i] == '0'&&b[i] == '1'))
			{
				if (falg)
				{
					ret.push_back('0');
					falg = true;
				}
				else
				{
					ret.push_back('1');
					falg = false;
				}
			}
			else if (a[i] == '0'&&b[i] == '0')
			{
				if (falg)
				{
					ret.push_back('1');
				}
				else
				{
					ret.push_back('0');
				}
				falg = false;
			}
		}
		
		int len_max = max(a.size(),b.size());
		int len_min = min(a.size(),b.size());
		for (int i = len_min; i < len_max; i++)
		{
			if (a.size() >= b.size())
			{
				if (a[i] == '1'&&falg)
				{

					ret.push_back('0');
					falg = true;

				}
				else if (a[i] == '0'&&falg)
				{

					ret.push_back('1');
					falg = false;
				}
				else
				{
					ret.push_back(a[i]);
				}
			}
			else
			{
				if (b[i] == '1'&&falg)
				{

					ret.push_back('0');
					falg = true;

				}
				else if (b[i] == '0'&&falg)
				{

					ret.push_back('1');
					falg = false;
				}
				else
				{
					ret.push_back(b[i]);
				}
			}
		}
		if (falg)
		{
			ret.push_back('1');
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}
};

// 65. Valid Number
class Solution_65 {
public:
	bool isNumber(string s) {
		int i = 0;
		int len = s.size();
		while (i<len&&s[i]==' ')
		{
			i++;
		}
		if (i == len)
		{
			return false;
		}
		while (i<len&&s[i] >= '0'&&s[i] <= '9')
		{
			i++;
		}
		if (i==len)
		{
			return true;
		}

		if (s[i] == '.')
		{
			i++;
			if (i==len)
			{
				return false;
			}
		}
		else if (s[i] == 'e')
		{
			if (i==0)
			{
				return false;
			}
			else
			{
				i++;
			}
		}
		else
		{
			while (i < len&&s[i] == ' ')
			{
				i++;
			}
			if (i == len)
			{
				return true;
			}
			return false;
		}
		
		
		while (i<len&&s[i] >= '0'&&s[i] <= '9')
			i++;
		
		while (i < len&&s[i] == ' ')
		{
			i++;
		}

		if (i==len)
		{
			return true;	
		}
		else
		{
			return false;
		}
	}

    //链接：https://www.nowcoder.com/questionTerminal/608d810765a34df2a0d47645626dd2d3
	class Solution {
	public:
		bool isNumber(const char *s)
		{
			string str(s);
			int index = str.find_first_not_of(' ');
			if (str[index] == '+' || str[index] == '-') //正负号
				index++;
			int points = 0, numbers = 0;
			for (; str[index] >= '0' && str[index] <= '9' || str[index] == '.'; index++) // 0.1 .1
				s[index] == '.' ? ++points : ++numbers;
			if (points > 1 || numbers < 1)
				return false;

			if (str[index] == 'e' || str[index] == 'E')
			{
				index++;
				if (str[index] == '+' || str[index] == '-') // E后面也有正负号
					index++;
				int afterE = 0;
				for (; str[index] >= '0' && str[index] <= '9'; index++)
					afterE++;
				if (afterE < 1)
					return false;
			}
			for (; str[index] == ' '; index++){}
			return str[index] == '\0';
		}
	};
};

class Solution_66 {
public:
	// 用一个数组表示一个整数的每一位，给这个整数+1，只有当前位为9时，才会有进位
	vector<int> plusOne(vector<int> &digits) {

		for (int i = digits.size() - 1; i >= 0; i--)
		{
			if (digits[i] == 9)
			{
				digits[i] = 0;
			}
			else
			{
				digits[i]++; //当前
				return digits;
			}
		}
		digits[0] = 1;
		digits.push_back(0); //多一位就行

		return digits;
	}
};

class Solution_68 {
public:
	vector<string> fullJustify(vector<string> &words, int L) {

	}
};

class Solution_69 {
public:
	int mySqrt(int x) {

		int ret = 0;
		int mx = INT_MAX;
		while (1)
		{
			long long temp = ret*ret;
			long long temp1 = (long long)(ret+1)*(long long)(ret+1); //关于数据的题考虑越界和边界条件；*操作之前强制类型转换
			if (temp1>mx)
			{
				return ret;
			}
			if (temp<= x && temp1>x) //考虑越界的问题
			{
				break;
			}
			ret++;
		}
		return ret;
	}

	int sqrt(int x) //牛顿逼近法
	{
		long r = x; 
		while (r*r > x)
			r = (r + x / r) / 2; 
		return r;
	}


	//特比特别要注意两点：第一right要取x / 2 + 1  这个还不是最重要的，其实只是影响速度
	//第二：要用x / middle > middle  来表示x > middle*middle  不然会溢出
	//第三：判断相等时用x / middle >= middle && x / (middle + 1) < (middle + 1)
	int sqrt_(int x)
	{
		if (x<2)
		{
			return x;
		}
		int l = 1, r = x/2+1;
		int mid=0;
		while (l<=r)
		{
			//mid = l + (r - l) / 2; 
			mid = l + ((r - l) >> 1); //位运算的优先级低于算术运算 bug: mid=l+(r-l)>>1
			if (mid != 0)
			{
				if (x / (mid + 1)<mid + 1 && x / mid>=mid)
				{
					return mid;
				}else if (x / mid>=mid) //不使用x>mid*mid
				{
					l = mid + 1;
				}
				else if(x / mid< mid)
				{
					r = mid - 1;
				}
			}
			
		}

		return mid;
	}

	int sqrt_ref(int x) {
		if (x == 0){
			return 0;
		}
		if (x < 0){
			return -1;
		}
		int left = 1, right = x / 2 + 1, middle;
		while (left <= right){
			middle = (left + right) / 2;
			if (x / middle >= middle && x / (middle + 1)<(middle + 1)){
				return middle;
			}
			else if (x / middle>middle){
				left = middle + 1;
			}
			else{
				right = middle - 1;
			}
		}
		return right;
	}
};


// add 70. Climbing Stairs
class Solution_70 {
public:
	int climbStairs(int n) {

		int ret = 0;
		int f1 = 1, f2 = 2;

		if (n == 1 || n == 2)
		{
			return n;
		}
		for (int i = 3; i <= n;i++)
		{
			int temp = f1 + f2;
			f1 = f2;
			f2 = temp;
		}
		return f2;
	}


	int climbStairs_(int n)
	{
		if (n==1||n==2)
		{
			return	n;
		}
		return climbStairs_(n - 1) + climbStairs_(n - 2);
	}

};

class Solution_71 {
public:

	// test:"/a/./b///../c/../././../d/..//../e/./f/./g/././//.//h///././/..///" output:"/e/f/g"
	string simplifyPath(string path) {

		stack<string> st;
		for (int i = 0; i < path.size();i++)
		{
			while (i < path.size() && path[i] == '/') i++;  //可能多个

			string str = "";
			while (i<path.size()&&path[i]!='/')  //记录'/'之间的字符串
			{
				str += path[i];
				i++;
			}
			if (str == ".")
			{
				continue;
			}
			if (str==".."&&!st.empty())
			{
				st.pop();
			}
			else if (str!=".."&&str!="") //必须有啊   /.. ; ///斜杆在末尾的时候str=""
			{
				st.push(str);
			}
		}
		string ret = "";
		if (st.empty())
		{
			return "/";
		}
		while (!st.empty())
		{
			ret = "/" + st.top()+ret; //加在后面，否则逆序
			st.pop();
		}

		return ret;
	}

};

class Backpack {
public:
	int maxValue(vector<int> w, vector<int> v, int n, int cap) {
		// write code here

		vector<vector<int>> dp(n + 1, vector<int>(cap + 1, 0));

		for (int i = 1; i <= n;i++)
		{
			for (int j = 1; j <= cap ;j++)
			{
				if (j>=w[i-1]) //w.v下标-1
				{
					dp[i][j] = max(dp[i-1][j],dp[i-1][j-w[i-1]]+v[i-1]);
				}
				else
				{
					dp[i][j] = dp[i-1][j];
				}
			}
		}
		return dp[n][cap];
	}

	int maxValue_(vector<int> w, vector<int> v, int n, int cap) {
		// write code here

		vector<int> dp(cap + 1, 0);
		vector<int> pre(cap + 1, 0); //记录上一行的dp值
		for (int i = 0; i < n;i++)
		{
			for (int j = 1; j <= cap;j++)
			{
				if (j>=w[i])
				{
					dp[j] = max(pre[j],pre[j-w[i]]+v[i] );
				}
				else
				{
					dp[j] = pre[j];
				}
			}
			pre = dp;
		}
		return dp[cap];
	}
};

// 72. Edit Distance
class Solution_72 {
public:
	//把问题转换为二维矩阵：
	//	arr[i][j]表示S1.sub(0, i)和S2.sub(0, j)的编辑距离，则
	//	arr[i][j] = min{ 1 + arr[i][j - 1], 1 + arr[i - 1][j], 1 + arr[i - 1][j - 1](当S1[i] != S2[j]), arr[i - 1][j - 1](当S1[i] == S2[j]) }
	//边界情况：arr[0][j] = j, arr[i][0] = i

	int minDistance(string word1, string word2) {

		if (word1.empty()&&word2.empty())
		{
			return 0;
		}
		int n = word1.size();
		int m = word2.size();

		vector<vector<int>> dp(n + 1, vector<int>(m+1, 0));

		for (int  i = 0; i <= n; i++)
		{
			dp[i][0] = i;
		}
		for (int j = 0; j <= m;j++)
		{
			dp[0][j] = j;
		}

		for (int i = 1; i <= n;i++)
		{
			for (int j = 1; j <= m;j++)
			{
				//S!-->S2; 当前匹配字符T1==T2,dp[i-1][j-1];不匹配时，删除T1,1+dp[i-1][j];增加T1,那么T1与j匹配，剩下i和j-1匹配,1+dp[i][j-1];更改T1,1+dp[i-1][j-1]
				if (word1[i-1]==word2[j-1])
				{
					dp[i][j] = dp[i - 1][j - 1];
				}
				else
				{
					dp[i][j] =min(1 + dp[i - 1][j - 1], min(1 + dp[i - 1][j], 1 + dp[i][j - 1]));
				}
				
			}
		}

		return dp[n][m];
	}
};

class Solution_73 {
public:
	void setZeroes(vector<vector<int> > &matrix) {
		
		int n = matrix.size();
		int m = matrix[0].size();
		
		bool row = false, col = false;
		//记录第一行，第一列是否有0
		for (int i = 0; i < n;i++)
		{
			if (matrix[i][0]==0)
			{
				row = true;
				break;
			}
		}
		for (int j = 0; j < m;j++)
		{
			if (matrix[0][j]==0)
			{
				col = true;
				break;
			}
		}

		//遍历其他位置，用第一行，第一列记录是否有0
		for (int i = 1; i < n;i++)
		{
			for (int j = 1; j < m;j++)
			{
				if (matrix[i][j]==0)
				{
					matrix[i][0] = 0;
					matrix[0][j] = 0;
				}
			}
		}

		//根据记录清0
		for (int i = 1; i < n;i++)
		{
			for (int j = 1; j < m;j++)
			{
				if (0==matrix[i][0]||0==matrix[0][j])
				{
					matrix[i][j] = 0;
				}
			}
		}

		// 处理第一行/列
		if (row)
		{
			for (int i = 0; i < n;i++)
			{
				matrix[i][0] = 0;
			}
		}
		if (col)
		{
			for (int j = 0; j < m;j++)
			{
				matrix[0][j] = 0;
			}
		}
		return;
	}
};

//74. Search a 2D Matrix
class Solution_74 {
public:
	bool searchMatrix(vector<vector<int>>& matrix, int target) {

		if (matrix.empty()||matrix[0].empty())
		{
			return false;
		}

		int n = matrix.size();
		int m = matrix[0].size();

		/*for (int i = n-1; i >= 0; ) //bug:导致每次循环边界条件判断有问题
		{
		for (int j = 0; j < m;)
		{*/
		int i = n - 1, j = 0;
		while (i>=0&&j<m)
		{
			{
				if (matrix[i][j]==target)
				{
					return true;
				}
				else if (matrix[i][j]<target)
				{
					j++;
				}
				else
				{
					--i;
				}
			}
		}

		return false;
	}
};

// 75. Sort Colors
class Solution_75 {
public:
	void sortColors(vector<int>& nums) {

		if (nums.size()<=1)
		{
			return;
		}
		int left = 0, right = nums.size() - 1; // //记录右边第一个非2的元素

		for (int i = 0; i <= right;i++)
		{
			if (nums[i]==0)
			{
				swap(nums[i],nums[left]);
				left++; //记录左边第一个非0的元素
			}else if (nums[i]==2)
			{
				swap(nums[i],nums[right]); //交换的数可能是0、1，需要重新判断，i不能++
				--i;
				right--;
			}
		}

		return;
	}

	void sortColors(int A[], int n) {

		//vector<int> vec(A, A + n); //传值没有出去
		//sortColors(vec);

		int *nums = A;
		if (n <= 1)
		{
			return;
		}
		int left = 0, right = n - 1; // //记录右边第一个非2的元素

		for (int i = 0; i <= right; i++)
		{
			if (nums[i] == 0)
			{
				swap(nums[i], nums[left]);
				left++; //记录左边第一个非0的元素
			}
			else if (nums[i] == 2)
			{
				swap(nums[i], nums[right]); //交换的数可能是0、1，需要重新判断，i不能++
				--i;
				right--;
			}
		}
		return;
	}
};

// 76. Minimum Window Substring
class Solution_76 {
public:

	//1) begin开始指向0， end一直后移，直到begin - end区间包含T中所有字符。记录窗口长度d
	//2) 然后begin开始后移移除元素，直到移除的字符是T中的字符则停止，此时T中有一个字符没被包含在窗口，
	//3) 继续后移end，直到T中的所有字符被包含在窗口，重新记录最小的窗口d。
	//4) 如此循环知道end到S中的最后一个字符。
	//时间复杂度为O(n)
	string minWindow_(string s, string t) {

		string result;
		if (s.empty()||s.size()<t.size())
		{
			return result;
		}
		unordered_map<char, int> mp; //存储t中的字符，便于与s匹配
		int left = 0;
		int cnt = 0;                 //窗口字符串进行计数
		int minlen = s.size()+1;
		for (int i = 0; i < t.size();i++)
		{
			mp[t[i]]++;
		}

		for (int right = 0; right < s.size(); right++)
		{
			if (mp.find(s[right])!=mp.end()) //t字符在left~right窗口
			{
				if (mp[s[right]]>0) //体会>0的作用：当s中有重复字符串时，第一次匹配
				{
					cnt++; //计数器+1
				}
				mp[s[right]]--; //有重复元素时候，可能减为负

				while (cnt==t.size()) ////当窗口内有t的所有字符，就要开始移动左窗口
				{
					if (mp.find(s[left])!=mp.end())
					{
						if (minlen>right-left+1)
						{
							minlen = right - left + 1;
							result = s.substr(left, right - left + 1);
						}
						mp[s[left]]++; //将其包含在mp内，继续查找
						if (mp[s[left]]>0)
						{
							cnt--;
						}
					}
					left++; //右移窗口左边
				}
			}
		}

		return result;
	}

	string minWindow(string S, string T) {

		return minWindow_(S, T);
	}
};

// 77. Combinations
class Solution_77 {
public:

	void help(vector<vector<int>>& vecs, vector<int> &vec,int i,int k,int n )
	{
		if (k==0)
		{
			vecs.push_back(vec);
			return;
		}
		if (i>n)
		{
			return;
		}

		vec.push_back(i + 1);
		help(vecs, vec, i + 1, k-1, n);
		vec.pop_back();
		help(vecs, vec, i + 1, k, n);

		return;

	}

	vector<vector<int>> combine(int n, int k) {

		vector<vector<int>> vecs;
		vector<int> vec;

		help(vecs,vec,0,k,n);

		return vecs;  
	}
};

// 78. Subsets
class Solution_78 {
public:
	void help(vector<vector<int>>& vecs, vector<int> &vec,vector<int> &src,int index,int k)
	{
		if (index>src.size())
		{
			return;
		}
		if (k==0)
		{
			vecs.push_back(vec);
			return;
		}

		vec.push_back(src[index]);
		help(vecs, vec, src, index + 1, k - 1);
		vec.pop_back();
		help(vecs, vec, src, index + 1, k);

		return;
	}

	vector<vector<int>> subsets(vector<int>& nums) {

		vector<vector<int>> vecs;
		vector<int> vec;

		sort(nums.begin(), nums.end());
		for (int i = 0; i <= nums.size();i++)
		{
			help(vecs, vec, nums, 0, i);
		}
		
		return vecs;
	}


    //链接：https://www.nowcoder.com/questionTerminal/c333d551eb6243e0b4d92e37a06fbfc9
	

	void backtracking(vector<vector<int>> &result, vector<int> &path, vector<int> &S, int n) {
		result.push_back(path);
		for (int i = n; i < S.size(); ++i) {
			path.push_back(S[i]);
			backtracking(result, path, S, i + 1);
			path.pop_back();
		}
	}
	vector<vector<int> > subsets_ref(vector<int> &S) {
		vector<vector<int>> result;
		vector<int> path;
		if (S.size() == 0) 
			return result;
		sort(S.begin(), S.end());
		backtracking(result, path, S, 0);

		return result;
	}

};


// 在类内部不能直接初始化变量
int ajd[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { -1, 0 }, { 1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 } }; //上下左右，四连通域

// 79. Word Search
class Solution_79 {
public:

	typedef pair<int, int> pii;

	bool judgeValid_bug(vector<vector<char>>& board, string word,int index, int x, int y)
	{
		int m = board.size();
		int n = board[0].size();

		pii p(x,y);
		queue<pii> que;
		que.push(p);

		if (index == word.size())
		{
			return true;
		}
		while (!que.empty())
		{
			int size = que.size();
			while (size--)
			{
				pii temp = que.front();
				x = temp.first;
				y = temp.second;
				que.pop();

				for (int i = 0; i < 8; i++)
				{
					if (ajd[i][0] + x >= 0 && (ajd[i][0] + x)<m && (ajd[i][1] + y) >= 0 && (ajd[i][1] + y)<n)
					{
						if (board[ajd[i][0] + x][ajd[i][1] + y] == word[index]) //未被归域下，才放入
						{
							temp.first = ajd[i][0] + x;
							temp.second = ajd[i][1] + y;
							que.push(temp);
							if (index == word.size() - 1)
							{
								return true;
							}
						}
					}
				}

			}
			index++;
		}
		return false;
	}

	// 上下左右 dfs
	bool judgeValid(vector<vector<char>>& board, string word, int index, int x, int y)
	{
		int m = board.size();
		int n = board[0].size();

		if (index >= word.size())
		{
			return true;
		}
		if (x < 0 || y < 0 || x >= m || y >= n)
		{
			return false; //超出边界
		}

		if (board[x][y]!=word[index])
		{
			return false;
		}

		char temp = board[x][y];
		board[x][y] = '.';  //节约used[i][j] = true; 空间 //防止从同一位置开始，以后重复使用

		bool ret = judgeValid(board,word,index+1,x-1,y)||
			judgeValid(board,word,index+1,x,y-1)||
			judgeValid(board,word,index+1,x,y+1)||
			judgeValid(board,word,index+1,x+1,y);

		board[x][y] = temp;

		return ret;
	}

	bool exist(vector<vector<char>>& board, string word) {
		
		if (board.empty())
		{
			return false;
		}
		if (word.empty())
		{
			return true;
		}
		int m = board.size();
		int n = board[0].size();

		for (int i = 0; i < m;i++)
		{
			for (int j = 0; j < n;j++)
			{
				if (board[i][j]==word[0])
				{
					if (judgeValid(board,word,0,i,j))
					{
						return true;
					}
				}
			}
		}

		return false;
	}


};

// 80. Remove Duplicates from Sorted Array II
class Solution_80 {
public:
	int removeDuplicates_(vector<int>& nums) {

		if (nums.size()<=1)
		{
			return nums.size();
		}

		int len = 0;
		int start = 0, end = 0;

		for (int i = 1; i < nums.size();i++)
		{
			if (nums[i]==nums[i-1])
			{
				end++;
			}
			else
			{
				start = i;
				end = i;
			}
			if (end-start+1<=2)
			{
				nums[++len] = nums[i];
			}
		}
		
		return len+1;
	}

	int removeDuplicates(int A[], int n) {
		
		vector<int> vec(A, A + n); //vec传值不能达到A；
		return removeDuplicates_(vec);
	}

	int removeDuplicates_1(int A[], int n) {
		int *nums = A;
		if (n <= 1)
		{
			return n;
		}

		int len = 0;
		int start = 0, end = 0;

		for (int i = 1; i < n; i++)
		{
			if (nums[i] == nums[i - 1])
			{
				end++;
			}
			else
			{
				start = i;
				end = i;
			}
			if (end - start + 1 <= 2)
			{
				nums[++len] = nums[i];
			}
		}

		return len + 1;
	}


};

// 81. Search in Rotated Sorted Array II
class Solution_81 {
public:
	// The array may contain duplicates.
	bool search_(vector<int>& nums, int target) {

		if (nums.empty())
		{
			return false;
		}
		int low = 0, high = nums.size() - 1;
		int mid = 0;
		while (low<high)
		{
			mid = low + (high - low) / 2;
			if (nums[mid]==target)
			{
				return true;	
			}
			if (nums[mid]>nums[high]) // 前半部分有序；后半部分无序
			{
				if (nums[mid]>target&&nums[low]<=target)
				{
					high = mid;
				}
				else
				{
					low = mid + 1;
				}
			}
			else if (nums[mid]<nums[high]) // 后半部分有序
			{
				if (nums[mid]<target&&target<=nums[high])
				{
					low = mid+1;
				}
				else
				{
					high = mid;
				}
			}
			else
			{
				high--;
			}
		}
		return nums[low] == target ? true : false;
	}

	bool search(int A[], int n, int target) {
		vector<int> vec(A, A + n);
		return search_(vec, target);
	}
};

// 82. Remove Duplicates from Sorted List II
class Solution_82 {
public:
	ListNode* deleteDuplicates(ListNode* head) {

		if (!head||!head->next)
		{
			return head;
		}

		ListNode*newHead = new ListNode(0);
		newHead->next = head;

		ListNode* pre = newHead;
		ListNode* cur = head;
		
		while (cur&&cur->next)
		{
			ListNode* next = cur->next;
		
			if(next->val!=cur->val)
			{
				if (pre->next==cur) //pre->next当前元素开始，cur当前元素结束，cur->next另外不同的元素
				{
					pre = cur;
				}
				else
				{
					pre->next = cur->next;
				}
			}
			cur = cur->next;
		}
		if (pre->next!=cur) //这里是地址比较，若没有重复元素，则地址相同的
		{
			pre->next = cur->next;
		}
		return newHead->next;
	}
};

class Solution_83 {
public:
	ListNode *deleteDuplicates(ListNode *head) {

		if (!head||!head->next)
		{
			return head;
		}

		ListNode* cur = head;
		ListNode*pre = NULL;
		while (cur&&cur->next)
		{
			pre = cur;
			cur = cur->next;
			ListNode* temp = pre; //记录每次重复点的开始位置
		    while(cur&&pre->val==cur->val)
			{
				pre = cur;
				cur=cur->next;
			}
			temp->next = cur; //跳过重复位置
		}
		return head;
	}
};

// 84. Largest Rectangle in Histogram
class Solution_84 {
public:
	int largestRectangleArea(vector<int>& heights) {

		int res = 0;
		stack<int> st; //存储递增的下标
		for (int i = 0; i < heights.size();i++)
		{
			while (!st.empty() && heights[st.top()]>heights[i]) //出栈操作，之前都是递增的
			{
				int h = heights[st.top()];
				st.pop();

				if (st.empty())
				{
					res = max(res, h*i);
				}
				else
				{
					res = max(res, h*(i - st.top()-1)); //当前区间[st.top+1,i-1]
				}
				
			}
			st.push(i);
		}

		while (!st.empty()) //递增的
		{
			int h = heights[st.top()];
			st.pop();
			int s = h * (st.empty() ? heights.size() : (heights.size() - st.top() - 1));
			res = max(res, s);
		}

		return res;
	}
};

// 85. Maximal Rectangle
class Solution_85 {
public:

	int largestRectangleArea(vector<int> &height) {
		int res = 0;
		stack<int> s;
		height.push_back(0);
		for (int i = 0; i < height.size(); ++i) {
			if (s.empty() || height[s.top()] <= height[i]) s.push(i);
			else {
				int tmp = s.top();
				s.pop();
				res = max(res, height[tmp] * (s.empty() ? i : (i - s.top() - 1)));
				--i;
			}
		}
		return res;
	}

	int maximalRectangle(vector<vector<char>>& matrix) {

		if (matrix.empty())
		{
			return 0;
		}
		int res = 0;

		int n = matrix.size();
		int m = matrix[0].size();

		vector<int> height(m);
		for (int i = 0; i < n;i++)
		{
			for (int j = 0; j < m;j++)
			{
				height[j] = matrix[i][j] == '0' ? 0 : (1+height[j]);
			}
			res =max(res, largestRectangleArea(height));
		}

		return res;
	}
};

// 86. Partition List
class Solution_86 {
public:

	//思路：新建两个节点preHead1与preHead2，分别为指向两个链表的头结点。

	//	把节点值小于x的节点链接到链表1上，节点值大等于x的节点链接到链表2上。
	//	最后把两个链表相连即可
	ListNode* partition(ListNode* head, int x) {

		if (!head||!head->next)
		{
			return head;
		}
	
		ListNode*cur = head;
		ListNode*left = new ListNode(0);
		ListNode*p = left;
		ListNode*right = new ListNode(0);
		ListNode*q = right;
		while (cur)
		{
			ListNode* temp = cur;	
			cur = cur->next;
			temp->next = NULL;

			if (temp->val<x)
			{
				left->next = temp;
				left = left->next;
			}
			else
			{
				right->next = temp;
				right = right->next;
			}
		}

		left->next = q->next;
		return p->next;
	}
};

// 87. Scramble String
class Solution_87 {
public:
	bool isScramble(string s1, string s2) {

		if (s1==s2)
		{
			return true;
		}
		if (s1.size()!=s2.size())
		{
			return false;
		}

		vector<int> hash(26,0);
		for (int i = 0; i < s1.size();i++)
		{
			hash[s1[i] - 'a']++;
			hash[s2[i] - 'a']--;
		}
		for (int i = 0; i < 26;i++)  //递归剪枝
		{
			if (hash[i]!=0)
			{
				return false;
			}
		}

		bool res = false;
		for (int i = 1; i < s1.size();i++) //遍历所有可能割开的位置, 切割的长度
		{
			res = res || (isScramble(s1.substr(0, i), s2.substr(0, i)) && isScramble(s1.substr(i, s1.size() - i), s2.substr(i, s1.size() - i)));   //长度要一致
			res = res || (isScramble(s1.substr(0, i), s2.substr(s1.size() - i)) && isScramble(s1.substr(i),s2.substr(0, s1.size()-i)));
		}

		return res;

	}
};

class Solution_88 {
public:
	// merge nums2 into nums1 as one sorted array.
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {

		if (m==0)
		{
			for (int i = 0; i < n;i++)
			{
				nums1.push_back(nums2[i]);
			}
			return;
		}
		if (n==0)
		{
			return;
		}
		int len1 = m - 1;
		int len2 = n - 1;
		int index = m + n - 1;
		while (len1>=0&&len2>=0)
		{
			if (nums2[len2]>nums1[len1])
			{
				nums1[index--] = nums2[len2--];
			}
			else
			{
				nums1[index--] = nums1[len1--];
			}
		}

		while (len2>=0)
		{
			nums1[index--] = nums2[len2--];
		}

		return;
	}

	void merge(int A[], int m, int B[], int n) {
		int i = m - 1;
		int j = n - 1;
		int index = m + n - 1;

		while (i>=0&&j>=0)
		{
			A[index--] = (A[i] > B[j]) ? A[i--] : B[j--];
		}
		while (j>=0)
		{
			A[index--] = B[j--];
		}
		return;
	}
};

// 89. Gray Code
class Solution_89 {
public:
	vector<int> grayCode(int n) {

	}
};

// 91. Decode Ways
class Solution_91 {
public:
	// 限制条件，比如说一位数时不能为0，两位数不能大于26，其十位上的数也不能为0
	int numDecodings(string s) {
		if (s.empty() || (s.size() > 1 && s[0] == '0'))
			return 0;
		vector<int> dp(s.size()+1,0); //表示前i个字符的解码方式
		dp[0] = 1;  
		for (int i = 1; i <= s.size();i++)
		{
			if (s[i-1]!='0') //i-1位置不为0，可以独立一种出来
			{
				dp[i] += dp[i - 1];
			}
			if (i>1 && (s[i - 2] == '1' || (s[i - 2] == '2'&& s[i - 1]>='0'&& s[i - 1] <= '6'))) //根据i-2的位置
			{
				dp[i] += dp[i - 2];
			}
		}
		return dp[s.size()];
	}

};


// 90. Subsets II
class Solution_90 {
public:

	void dfs(vector<vector<int>> &res,vector<int> &out,vector<int>&nums,int pos)
	{
		res.push_back(out);

		for (int i = pos; i < nums.size();i++)
		{
			out.push_back(nums[i]);
			dfs(res, out, nums, i + 1);
			out.pop_back();
			while ((i+1)<nums.size()&&nums[i+1]==nums[i])
			{
				i++;
			}
		}
		return;
	}

	vector<vector<int>> subsetsWithDup(vector<int>& nums) {

		vector<vector<int>> res;
		vector<int> out;
		if (nums.empty())
		{
			return res;
		}
		sort(nums.begin(),nums.end());

		dfs(res, out, nums, 0);

		return res;

	}
};


class Solution_92 {
public:
	ListNode* reverseBetween(ListNode* head, int m, int n) {

		ListNode* newHead = new ListNode(0);
		newHead->next = head;

		ListNode* pre=NULL, *cur=newHead, *front=NULL;

		for (int i = 0; i < m - 1;i++)
		{
			cur = cur->next;
		}
		pre = cur;       //记录反转之前的节点
		ListNode* last = cur->next; //也是反转后的尾指针

		for (int i = m; i <= n;i++)
		{
			cur = pre->next;
			pre->next = cur->next;
			cur->next = front; //向front节点前插入,front每次前移
			front = cur;
		}

		cur = pre->next;
		pre->next = front;
		last->next = cur;

		return newHead->next;
	}
};

class Solution_99 {
public:
	void recoverTree(TreeNode* root) {

		TreeNode* first = NULL, *second = NULL, *parenet = NULL;
		TreeNode*cur, *pre; //中序遍历
		cur = root;

		while (cur)
		{
			if (cur->left==NULL)
			{
				if (parenet&&parenet->val>cur->val)
				{
					if (!first)
					{
						first = parenet;
					}
					second = cur;
				}
				parenet = cur;
				cur = cur->right;
			}
			else
			{
				pre = cur->left; //找到左子树的最右节点
				while (pre->right&&pre->right!=cur)
				{
					pre = pre->right;
				}

				if (!pre->right)
				{
					pre->right = cur;
					cur = cur->left;
				}
				else
				{
					pre->right = NULL; //恢复树结构
					if (parenet&&parenet->val > cur->val)
					{
						if (!first)
						{
							first = parenet;
						}
						second = cur;
					}
					parenet = cur;
					cur = cur->right;
				}
			}
		}
		if (first&&second)
		{
			swap(first->val, second->val);
		}
	}
};

class Solution_98 {
public:

	//bug
	bool isValidBST_bug(TreeNode* root) {

		if (!root||(!root->right&&!root->left))
		{
			return true;
		}
		
		if (root->left!=NULL&&root->left->val>=root->val)
		{
			return false;
		}
		if (root->right!=NULL&&root->right->val<=root->val)
		{
			return false;
		}
		return isValidBST_bug(root->left) && isValidBST_bug(root->right);
	}

	// 二分查找树的中序遍历结果是一个递增序列
	TreeNode* pre = NULL;
	void InOrder(TreeNode* root,int &res)
	{
		if (!root)
		{
			return;
		}
		InOrder(root->left, res);
		if (!pre)
		{
			pre = root;
		}
		else
		{
			if (root->val<=pre->val)
			{
				res = 0;
			}
			pre = root;
		}

		InOrder(root->right,res);
		return;
	}
	bool isValidBST(TreeNode *root) {

		if (!root)
		{
			return true;
		}
		int res = 1;
		InOrder(root,res);

		if (res==0)
		{
			return false;
		}
		return true;
	}
};


class Solution_94 {
public:
	vector<int> inorderTraversal(TreeNode* root) {

		vector<int> res;
		if (root==NULL)
		{
			return res;
		}
		stack<TreeNode*> st;
		TreeNode* cur = root;
		while (!st.empty()||cur)
		{
			while (cur)
			{
				st.push(cur);
				cur = cur->left;
			}
			if (!st.empty())
			{
				cur = st.top();
				res.push_back(cur->val);
				st.pop();
				cur = cur->right;
			}
		}
		return res;
	}
};


// 96. Unique Binary Search Trees
class Solution_96 {
public:
	int numTrees(int n) {
		//卡特兰数
		long long ans = 1;
		for (int i = n + 1; i <= 2 * n;i++)
		{
			ans = ans*i / (i-n);
		}
		return ans / (n + 1);
	}
};


class Solution_95 {
public:
	vector<TreeNode *> generateTrees(int n) {

		vector<TreeNode *> ves = GenerateSubTree(1, n + 1);

		return ves;
	}

	vector<TreeNode*> GenerateSubTree(int l, int r) {
		vector<TreeNode *> subTree;

		if (l >= r) {
			subTree.push_back(NULL);
			return subTree;
		}

		if (l == r - 1) {
			subTree.push_back(new TreeNode(l));
			return subTree;
		}


		for (int i = l; i < r; ++i) {
			vector<TreeNode *> leftSubTree = GenerateSubTree(l, i);
			vector<TreeNode *> rightSubTree = GenerateSubTree(i + 1, r);

			for (int m = 0; m < leftSubTree.size(); ++m) {
				for (int n = 0; n < rightSubTree.size(); ++n) {
					TreeNode *root = new TreeNode(i);
					root->left = leftSubTree[m];
					root->right = rightSubTree[n];
					subTree.push_back(root);
				}
			}
		}

		return subTree;
	}
};


class Solution_93 {
public:
	bool isValid(string str)
	{
		long temp = atol(str.c_str()); //用int溢出；atol
		if (temp>255)
		{
			return false;
		}
		if (str[0]=='0'&&str.size()>1)
		{
			return false;
		}
		return true;
	}
	void dfs(vector<string> &res, string t, string s, int cnt)
	{
		if (cnt>3)
		{
			return;
		}
		if (cnt==3&&isValid(s)) //最后一组数
		{
			res.push_back(t+s);
			return;
		}
		for (int i = 1; i < 4 && i < s.size();i++)
		{
			string sub = s.substr(0, i);
			if (isValid(sub))
			{
				dfs(res, t + sub + '.', s.substr(i), cnt + 1);
			}
		}
		return;
	}

	vector<string> restoreIpAddresses(string s) {

		vector<string> res;
		string t;
		dfs(res,t,s,0);
		return res;
	}
};


class Solution_97 {
public:
	bool isInterleave(string s1, string s2, string s3) {

		if (s3.length() != s1.length() + s2.length())
			return false;

		//bool table[s1.length() + 1][s2.length() + 1];
		vector<vector<bool>> table(s1.length() + 1, vector<bool>(s2.length() + 1, false));

		for (int i = 0; i < s1.length() + 1; i++)
		for (int j = 0; j < s2.length() + 1; j++){
			if (i == 0 && j == 0)
				table[i][j] = true;
			else if (i == 0)
				table[i][j] = (table[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
			else if (j == 0)
				table[i][j] = (table[i - 1][j] && s1[i - 1] == s3[i + j - 1]);
			else
				table[i][j] = (table[i - 1][j] && s1[i - 1] == s3[i + j - 1]) || (table[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
		}

		return table[s1.length()][s2.length()];
	}
};

class Solution_236 {
public:
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {

		if (root==NULL||root==q||root==p)
		{
			return root;
		}

		TreeNode* left = lowestCommonAncestor(root->left, p, q);
		TreeNode* right = lowestCommonAncestor(root->right, p, q);

		if (left!=NULL&&right!=NULL)
		{
			return root;
		}

		return left == NULL ? right : left;
	}
};




class LRUCache {
public:
	// 定义数据结构
	int n;
	list<pair<int, int>> lis; //链表头为最新访问的，链表尾为最近最少访问的
	map<int, list<pair<int, int>>::iterator> mp; //记录每个关键字在list中的位置

	LRUCache(int capacity) {
		n = capacity;
	}

	int get(int key) {

		int ret = -1;
		if (mp.find(key) != mp.end())  //已经存在；更新cache顺序就行
		{
			auto iter = mp[key];
			ret = iter->second; //记录查询key的value

			lis.erase(iter);
			lis.push_front(make_pair(key, ret));

			mp[key] = lis.begin();

		}
		return ret;
	}

	void put(int key, int value) {

		auto iter = mp.find(key); //iter是mp的迭代器

		if (iter != mp.end()) //已经存在，更新list位置
		{
			lis.erase(iter->second);
		}
		else if (lis.size() < n)
		{

		}
		else //list里面没有key，且list已经满了
		{
			//auto it = lis.end(); //end()迭代器指向最后元素的后面位置
			//it--;
			//m.erase(it->first);
			//lis.erase(it);

			int key = lis.back().first; //这样操作避免使用迭代器失效问题出现
			lis.pop_back();
			mp.erase(key);
		}

		lis.push_front(make_pair(key, value));
		mp[key] = lis.begin();
	}
};



#define cin infile
#include <fstream>
#include <iomanip>  //setprecision() setw()

//int main()
//{  
//
//	Solution_93 su_93;
//	su_93.restoreIpAddresses("2736786374048"); //2736786374048
//
//	Solution_96 su_96;
//	int ret_96=su_96.numTrees(19);
//
//	Solution_92 su_92;
//	list<int> list = {1,2,3,4,5};
//	ListNode* head_92=new ListNode(1);
//	head_92->next = new ListNode(2);
//	head_92->next->next = new ListNode(3);
//	head_92->next->next->next = new ListNode(4);
//	head_92->next->next->next->next = new ListNode(5);
//	su_92.reverseBetween(head_92, 2, 4);
//
//
//	Solution_90 su_90;
//	vector<int> vec_90 = {1,2,2};
//	su_90.subsetsWithDup(vec_90);
//
//	Solution_91 su_91;
//	su_91.numDecodings("0");
//
//	Solution_86 su_86;
//	ListNode*head_86 = new ListNode(1);
//	head_86->next = new ListNode(4);
//	head_86->next->next = new ListNode(2);
//	su_86.partition(head_86,3);
//
//	Solution_84 su_84;
//	vector<int> vec_84 = { 2, 1, 5, 6, 2, 3 };
//	su_84.largestRectangleArea(vec_84);
//
//	Solution_82 su_83;
//	ListNode*head_83 = new ListNode(1);
//	head_83->next = new ListNode(1);
//	head_83->next->next = new ListNode(3);
//	head_83->next->next->next = new ListNode(3);
//	su_83.deleteDuplicates(head_83);
//
//	Solution_80 su_80;
//	su_80.removeDuplicates_(vector<int>({1,1,1,2}));
//	int A_80[] = {1,1,1,2};
//	su_80.removeDuplicates(A_80, 4);
//
//	Solution_76 su_76;
//	su_76.minWindow("bba", "ab");
//
//	Solution_75 su_75;
//	int a_75[] = { 1, 0 };
//	su_75.sortColors(a_75,2);
//
//	Solution_74 su_74;
//	su_74.searchMatrix(vector<vector<int>>(1,vector<int>(1,1)),0);
//
//
//	Solution_72 su_72;
//	su_72.minDistance("b", "");
//
//	Backpack b;
//	int bbbb=b.maxValue_(vector<int>({ 1, 2, 3 }),vector<int>({ 1, 2, 3 }), 3, 6);
//
//	Solution_71 su_71;
//	su_71.simplifyPath("/..");
//
//
//	Solution_69 su_69;
//	int ret_69 = su_69.sqrt_(46340 * 46340);
//
//	Solution_65 su_65;
//	su_65.isNumber(".1");
//
//	Solution_67 su_67;
//	su_67.addBinary("100", "110010");
//
//
//	Solution_62 su_62;
//	int ret_62=su_62.uniquePaths(3, 7);
//	int ret_62_=su_62.uniquePaths2(3, 7);
//
//	ListNode* head_61 = new ListNode(1);
//	head_61->next = new ListNode(2);
//	/*head_61->next->next = new ListNode(3);
//	head_61->next->next->next = new ListNode(4);
//	head_61->next->next->next->next = new ListNode(5);*/
//	Solution_61 su_61;
//	su_61.rotateRight(head_61,2);
//
//
//	Solution_60 su_60;
//	su_60.getPermutation_ref(2,1);
//
//	Solution_54 su_54;
//	//vector<vector<int>> vec(3, vector<int>(3, 1));
//	su_54.spiralOrder(vector<vector<int>>(0, vector<int>(3, 1)));
//
//
//	Solution_51 su_51;
//	su_51.solveNQueens(4);
//
//
//	Solution_50 su_50;       
//	su_50.myPow1(2.1, 3);
//
//
//	Solution_49 su_49;
//	//su_49.anagrams(vector<string>(""));
//
//	vector<int> vec_test;
//	for (int i = 0; i < 10;i++)
//	{
//		vec_test.push_back(i);
//	}
//
//	for (vector<int>::iterator iter = vec_test.begin(); iter != vec_test.end(); iter++)
//	{
//		cout << *iter<<" "; //迭代器类指针
//	}
//
//	for (auto v:vec_test)
//	{
//		cout << v << " "; //直接输出元素
//	}
//
//	Solution_43 su_43;
//	string str_43 =su_43.multiply("98", "9");
//
//
//	Solution_42 su_42;
//	vector<int> vec_42;
//	su_42.trap(vec_42);
//
//
//	int a_37[][2] = { { 1, 2 }, {1,2} };
//	char A_37[][9] = { 
//		{ '.', '.', '9', '7', '4', '8', '.', '.', '.' },
//		{ '7', '.', '.', '.', '.', '.', '.', '.', '.' },
//		{ '.', '2', '.', '1', '.', '9', '.', '.', '.' },
//		{ '.', '.', '7', '.', '.', '.', '2', '4', '.' },
//		{ '.', '6', '4', '.', '1', '.', '5', '9', '.' },
//		{ '.', '9', '8', '.', '.', '.', '3', '.', '.' },
//		{ '.', '.', '.', '8', '.', '3', '.', '2', '.' },
//		{ '.', '.', '.', '.', '.', '.', '.', '.', '6' },
//		{ '.', '.', '.', '2', '7', '5', '9', '.', '.' }
//	};				
//	vector<vector<char>> vec_37;
//	for (int i = 0; i < 9;i++)
//	{
//		vector<char> tem_37;
//		for (int j = 0; j < 9;j++)
//		{
//			tem_37.push_back(A_37[i][j]);
//		}
//		vec_37.push_back(tem_37);
//	}
//	Solution_37 su_37;
//	su_37.solveSudoku(vec_37);
//
//	Solution_34 su_34;
//	su_34.searchRange(vector<int>({ 5, 7, 7, 8, 8, 10 }), 8);
//	int A_34[] = {1,1};
//	su_34.searchRange2(A_34, 2, 1);
//
//
//	Solution_33 su_33;
//	int A_33[] = { 3,1 };
//	int ret_33=su_33.search(vector	<int>({3,1}), 1);
//
//	Solution_32 su_32;
//	su_32.longestValidParentheses("(())()");
//
//	Solution_46 su_46;
//	su_46.permute(vector<int>({1,2,3}));
//
//	Solution_28 su_28;
//	su_28.strStr("banananobano", "nano");
//
//	ListNode* a_1 = new ListNode(1);
//	ListNode* a_2 = new ListNode(2);
//	ListNode* a_3 = new ListNode(3);
//	ListNode* a_4 = new ListNode(4);
//	ListNode* a_5 = new ListNode(5);
//	a_1->next = a_2; a_2->next = a_3; a_3->next = a_4; a_4->next = a_5;
//	Solution_25 su_25;
//	su_25.reverseKGroup(a_1,2);
//
//
//	Solution_14 su_14;
//	su_14.longestCommonPrefix(vector<string>());
//
//	Solution_13 su_13;
//	su_13.romanToInt("DCXXI");
//
//	Solution_10 su_10;
//	su_10.isMatch("aab", "c*a*b");
//
//	Solution_8 su_8;
//	su_8.myAtoi("-1");
//
//	//C++文件输入
//	ifstream infile("in.txt", ifstream::in);
//	vector<vector<int> > triangle;
//	int n_;
//	cin >> n_;
//	for (int i = 0; i < n_; i++) {
//		vector<int> vi(i + 1, 0);
//		for (int j = 0; j < i + 1; j++)
//			cin >> vi[j];
//		triangle.push_back(vi);
//	}
//	Solution_120* s_120 = new Solution_120();
//	cout << s_120->minimumTotal(triangle) << endl;
//	return 0;
//	
//	//c文件输入
//	freopen("in.txt", "r", stdin);
//    int ans = 0;
//	cin >> n_;
//	for (int i = 0; i < n_; i++){
//		for (int j = 0; j < n_; j++){
//			int x; 
//			scanf("%d", &x);
//			ans += x;
//		}
//	}
//	cout << ans << endl;
//	return 0;
//
//
//	string str_3 = "abba"; //"dvdf"  pwwkew "abba"
//	Solution_3 su_3;
//	su_3.lengthOfLongestSubstring(str_3);
//
//
//	vector<int> vec1(10, 1); //声明一个初始大小为10且值都是1的向量
//	vector<int> tmp(vec1.begin(), vec1.begin() + 3);  //用向量vec的第0个到第2个值初始化tmp
//
//	vector<vector<int>> ret;
//	int curCount = 1;
//	ret.push_back(vector<int>(0, curCount));
//
//	char str3[13];
//	scanf("%s\n", str3);
//	//how are you?
//	printf("%s\n", str3);
//
//	str3[1] = 'c'; //字符的赋值
//
//	char str1[] = "how are you?";
//	printf("%s\n", str1);
//	
//	char str2[20];
//	gets(str2); //how are you?
//	puts(str2);
//
//	char ch[] = "china\0china";
//	cout << sizeof(ch) << endl;
//	cout << strlen(ch) << endl;
//
//	//test01
//	char*a[] = { "work", "at", "alibaba" };
//	char**pa = a;
//	pa++;
//	printf("%s", *pa);
//
//	//cout << sizeof(bu) << endl;
//
//	//  可以做一个实验：string a = "aaaaaa";a[0] = '\0';a[1] = '\0';cout<<a;你会发现string确实可以存储多个\0.
//	//  但是不可以string a = "\0aaaa";这样是一个\0也不会存储的，因为a.capacity()结果为0.
//	string str_temp = "china\0\0china\0\0";
//
//	double  num = 4.5;
//	int num1 = 6;
//
//	Solution_119 su_119;
//	su_119.getRow(4);
//
//
//	int n[] = { 1, 4, 2, 3, 5, 0 };
//	vector<int>v(n, n + sizeof(n) / sizeof(int));//sizeof(n)/sizeof(int)是求数组n的长度  
//	cout << *min_element(v.begin(), v.end()) << endl;//最小元素  
//	cout << *max_element(v.begin(), v.end()) << endl;//最大元素  
//	return 0;
//
//	string begin = "a";
//	string end = "c";
//	unordered_set<string> dict;
//	dict.insert("a");
//	dict.insert("b");
//	dict.insert("c");
//	Solution_127_old su_127;
//	su_127.ladderLength(begin, end, dict);
//
//
//	vector<vector<char>> vec;
//
//	vector<char> temp(4,'X');
//	vec.push_back(temp);
//	char tem1[] = {'X','X','O','X'};
//	vector<char> temp1(tem1, tem1 + 4);
//	vec.push_back(temp1);
//	char tem2[] = { 'X', 'O', 'X', 'X' };
//	vector<char> temp2(tem2, tem2 + 4);
//	vec.push_back(temp2);
//
//	char tem3[] = { 'X', 'O', 'X', 'X' };
//	vector<char> temp3(tem3, tem3 + 4);
//	vec.push_back(temp3);
//
//	Solution_130 su_130;
//	su_130.solve(vec);
//
//
//	Solution_132 su_132;
//	string str_132 = "ab";
//	su_132.minCut(str_132);
//
//	Solution_131 su_131; 
//	string str = "aab";
//	su_131.partition(str);
//
//	//vector<int> vec;
//	//vec.push_back(2);
//    //vec.push_back(2);
//
//	//Solution_135 su_135;
//	//int ret_135=su_135.candy(vec);
//
//	RandomListNode *r_head = new RandomListNode(-1);
//
//	RandomListNode* next = new RandomListNode(-1);
//	r_head->next = next;
//
//	Solution_138 su_138;
//	RandomListNode* ret_138=su_138.copyRandomList(r_head);    
//
//
//
//	vector<int>array;
//	array.push_back(100);
//	array.push_back(300);
//	array.push_back(300);
//	array.push_back(300);
//	array.push_back(300);
//	array.push_back(500);
//	vector<int>::iterator itor;
//	for (itor = array.begin(); itor != array.end(); itor++)
//	{
//		cout << "array.size()="<<array.size() << endl;
//		if (*itor == 300)
//		{
//			itor = array.erase(itor);  //避免迭代器失效
//		}
//	}
//	for (itor = array.begin(); itor != array.end(); itor++)
//	{
//		cout << *itor << "";
//	}
//
//	// test list
//	ListNode* head = new ListNode(1);
//	ListNode* node2= new ListNode(2);
//	head->next = node2;
//
//	//ListNode* node3 = new ListNode(3);
//	//node2->next = node3;
//
//	//ListNode* node4 = new ListNode(4);
//	//node3->next = node4;
//
//	Solution_142 su142;
//	ListNode* ret142=su142.detectCycle(head);
//
//	Solution_143 su;
//	su.reorderList(head);
//
//	Solution_148 su148;
//	ListNode* ret_148 = su148.sortList(head);
//
//
//
//	
//	//test tree
//
//	return 0;
//}



void prim(int m, int n) //分解质因数
{
	if (m >= n)
	{
		while (m%n) n++;
		m /= n;
		prim(m, n);
		cout << n << endl;
	}
}

bool wordBreak(string s, unordered_set<string> &dict) {
	// BFS
	queue<int> BFS;
	unordered_set<int> visited;

	BFS.push(0);
	while (BFS.size() > 0)
	{
		int start = BFS.front();
		BFS.pop();
		if (visited.find(start) == visited.end()) //没有找到
		{
			visited.insert(start);
			for (int j = start; j < s.size(); j++)
			{
				string word(s, start, j - start + 1);
				if (dict.find(word) != dict.end()) //找到了子元素
				{
					BFS.push(j + 1);
					if (j + 1 == s.size())
						return true;
				}
			}
		}
	}

	return false;
}

bool pointInPolygon(int polySides, int polyY[],int polyX[],int x,int y) {

	int   i, j = polySides - 1;
	bool  oddNodes = false;

	for (i = 0; i < polySides; i++) {
		if (polyY[i] < y && polyY[j] >= y|| polyY[j] < y&& polyY[i] >= y) {

			if (polyX[i] + (y - polyY[i]) / (polyY[j] - polyY[i])*(polyX[j] - polyX[i]) < x) {

				oddNodes = !oddNodes;
			}
		}
		j = i;
	}

	return oddNodes;
}