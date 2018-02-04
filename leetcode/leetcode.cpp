
#include<iostream>
#include<math.h>

#include <vector>
#include<string>
#include<deque>
#include <stack>
#include <queue>
#include<map>
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

#define cin infile
#include <fstream>
#include <iomanip>  //setprecision() setw()
int main()
{   
	int a_37[][2] = { { 1, 2 }, {1,2} };
	char A_37[][9] = { 
		{ '.', '.', '9', '7', '4', '8', '.', '.', '.' },
		{ '7', '.', '.', '.', '.', '.', '.', '.', '.' },
		{ '.', '2', '.', '1', '.', '9', '.', '.', '.' },
		{ '.', '.', '7', '.', '.', '.', '2', '4', '.' },
		{ '.', '6', '4', '.', '1', '.', '5', '9', '.' },
		{ '.', '9', '8', '.', '.', '.', '3', '.', '.' },
		{ '.', '.', '.', '8', '.', '3', '.', '2', '.' },
		{ '.', '.', '.', '.', '.', '.', '.', '.', '6' },
		{ '.', '.', '.', '2', '7', '5', '9', '.', '.' }
	};				
	vector<vector<char>> vec_37;
	for (int i = 0; i < 9;i++)
	{
		vector<char> tem_37;
		for (int j = 0; j < 9;j++)
		{
			tem_37.push_back(A_37[i][j]);
		}
		vec_37.push_back(tem_37);
	}
	Solution_37 su_37;
	su_37.solveSudoku(vec_37);

	Solution_34 su_34;
	su_34.searchRange(vector<int>({ 5, 7, 7, 8, 8, 10 }), 8);
	int A_34[] = {1,1};
	su_34.searchRange2(A_34, 2, 1);


	Solution_33 su_33;
	int A_33[] = { 3,1 };
	int ret_33=su_33.search(vector	<int>({3,1}), 1);

	Solution_32 su_32;
	su_32.longestValidParentheses("(())()");

	Solution_46 su_46;
	su_46.permute(vector<int>({1,2,3}));

	Solution_28 su_28;
	su_28.strStr("banananobano", "nano");

	ListNode* a_1 = new ListNode(1);
	ListNode* a_2 = new ListNode(2);
	ListNode* a_3 = new ListNode(3);
	ListNode* a_4 = new ListNode(4);
	ListNode* a_5 = new ListNode(5);
	a_1->next = a_2; a_2->next = a_3; a_3->next = a_4; a_4->next = a_5;
	Solution_25 su_25;
	su_25.reverseKGroup(a_1,2);


	Solution_14 su_14;
	su_14.longestCommonPrefix(vector<string>());

	Solution_13 su_13;
	su_13.romanToInt("DCXXI");

	Solution_10 su_10;
	su_10.isMatch("aab", "c*a*b");

	Solution_8 su_8;
	su_8.myAtoi("-1");

	//C++文件输入
	ifstream infile("in.txt", ifstream::in);
	vector<vector<int> > triangle;
	int n_;
	cin >> n_;
	for (int i = 0; i < n_; i++) {
		vector<int> vi(i + 1, 0);
		for (int j = 0; j < i + 1; j++)
			cin >> vi[j];
		triangle.push_back(vi);
	}
	Solution_120* s_120 = new Solution_120();
	cout << s_120->minimumTotal(triangle) << endl;
	return 0;
	
	//c文件输入
	freopen("in.txt", "r", stdin);
    int ans = 0;
	cin >> n_;
	for (int i = 0; i < n_; i++){
		for (int j = 0; j < n_; j++){
			int x; 
			scanf("%d", &x);
			ans += x;
		}
	}
	cout << ans << endl;
	return 0;


	string str_3 = "abba"; //"dvdf"  pwwkew "abba"
	Solution_3 su_3;
	su_3.lengthOfLongestSubstring(str_3);


	vector<int> vec1(10, 1); //声明一个初始大小为10且值都是1的向量
	vector<int> tmp(vec1.begin(), vec1.begin() + 3);  //用向量vec的第0个到第2个值初始化tmp

	vector<vector<int>> ret;
	int curCount = 1;
	ret.push_back(vector<int>(0, curCount));

	char str3[13];
	scanf("%s\n", str3);
	//how are you?
	printf("%s\n", str3);

	str3[1] = 'c'; //字符的赋值

	char str1[] = "how are you?";
	printf("%s\n", str1);
	
	char str2[20];
	gets(str2); //how are you?
	puts(str2);

	char ch[] = "china\0china";
	cout << sizeof(ch) << endl;
	cout << strlen(ch) << endl;

	//test01
	char*a[] = { "work", "at", "alibaba" };
	char**pa = a;
	pa++;
	printf("%s", *pa);

	//cout << sizeof(bu) << endl;

	//  可以做一个实验：string a = "aaaaaa";a[0] = '\0';a[1] = '\0';cout<<a;你会发现string确实可以存储多个\0.
	//  但是不可以string a = "\0aaaa";这样是一个\0也不会存储的，因为a.capacity()结果为0.
	string str_temp = "china\0\0china\0\0";

	double  num = 4.5;
	int num1 = 6;

	Solution_119 su_119;
	su_119.getRow(4);


	int n[] = { 1, 4, 2, 3, 5, 0 };
	vector<int>v(n, n + sizeof(n) / sizeof(int));//sizeof(n)/sizeof(int)是求数组n的长度  
	cout << *min_element(v.begin(), v.end()) << endl;//最小元素  
	cout << *max_element(v.begin(), v.end()) << endl;//最大元素  
	return 0;

	string begin = "a";
	string end = "c";
	unordered_set<string> dict;
	dict.insert("a");
	dict.insert("b");
	dict.insert("c");
	Solution_127_old su_127;
	su_127.ladderLength(begin, end, dict);


	vector<vector<char>> vec;

	vector<char> temp(4,'X');
	vec.push_back(temp);
	char tem1[] = {'X','X','O','X'};
	vector<char> temp1(tem1, tem1 + 4);
	vec.push_back(temp1);
	char tem2[] = { 'X', 'O', 'X', 'X' };
	vector<char> temp2(tem2, tem2 + 4);
	vec.push_back(temp2);

	char tem3[] = { 'X', 'O', 'X', 'X' };
	vector<char> temp3(tem3, tem3 + 4);
	vec.push_back(temp3);

	Solution_130 su_130;
	su_130.solve(vec);


	Solution_132 su_132;
	string str_132 = "ab";
	su_132.minCut(str_132);

	Solution_131 su_131; 
	string str = "aab";
	su_131.partition(str);

	//vector<int> vec;
	//vec.push_back(2);
    //vec.push_back(2);

	//Solution_135 su_135;
	//int ret_135=su_135.candy(vec);

	RandomListNode *r_head = new RandomListNode(-1);

	RandomListNode* next = new RandomListNode(-1);
	r_head->next = next;

	Solution_138 su_138;
	RandomListNode* ret_138=su_138.copyRandomList(r_head);    



	vector<int>array;
	array.push_back(100);
	array.push_back(300);
	array.push_back(300);
	array.push_back(300);
	array.push_back(300);
	array.push_back(500);
	vector<int>::iterator itor;
	for (itor = array.begin(); itor != array.end(); itor++)
	{
		cout << "array.size()="<<array.size() << endl;
		if (*itor == 300)
		{
			itor = array.erase(itor);  //避免迭代器失效
		}
	}
	for (itor = array.begin(); itor != array.end(); itor++)
	{
		cout << *itor << "";
	}

	// test list
	ListNode* head = new ListNode(1);
	ListNode* node2= new ListNode(2);
	head->next = node2;

	//ListNode* node3 = new ListNode(3);
	//node2->next = node3;

	//ListNode* node4 = new ListNode(4);
	//node3->next = node4;

	Solution_142 su142;
	ListNode* ret142=su142.detectCycle(head);

	Solution_143 su;
	su.reorderList(head);

	Solution_148 su148;
	ListNode* ret_148 = su148.sortList(head);



	
	//test tree

	return 0;
}



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