
#include <iostream>
using namespace std;

int binarysearch(int array[], int low, int high, int target)
{
	if (low > high) return -1;

	int mid = (low + high) / 2;
	if (array[mid] > target)
		return    binarysearch(array, low, mid - 1, target);
	if (array[mid] < target)
		return    binarysearch(array, mid + 1, high, target);

	//if (midValue == target)
	return mid;
}

//Find the fisrt element, whose value is larger than target, in a sorted array 
int BSearchUpperBound(int array[], int low, int high, int target)
{
	//Array is empty or target is larger than any every element in array 
	if (low > high || target >= array[high]) return -1;

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

//Find the fisrt element, whose value is larger than target, in a sorted array 
int BSearchUpperBound_ext(int array[], int low, int high, int target)
{
	//Array is empty or target is larger than any every element in array 
	if (low > high || target > array[high]) return -1;

	int mid = (low + high) / 2;
	while (high > low)
	{
		if (array[mid] >= target)
			high = mid;
		else
			low = mid + 1;

		mid = (low + high) / 2;
	}

	return mid;
}

//Find the last element, whose value is less than target, in a sorted array 
int BSearchLowerBound(int array[], int low, int high, int target)
{
	//Array is empty or target is less than any every element in array
	if (high < low || target <= array[low]) return -1;

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

//Find the last element, whose value is less than target, in a sorted array 
int BSearchLowerBound_ext(int array[], int low, int high, int target)
{
	//Array is empty or target is less than any every element in array
	if (high < low || target < array[low]) return -1;

	int mid = (low + high + 1) / 2; //make mid lean to large side
	while (low < high)
	{
		if (array[mid] <= target)
			low = mid;
		else
			high = mid - 1;

		mid = (low + high + 1) / 2;
	}

	return mid;
}
//return type: pair<int, int>
//the fisrt value indicate the begining of range,
//the second value indicate the end of range.
//If target is not find, (-1,-1) will be returned
pair<int, int> SearchRange(int A[], int n, int target)
{
	pair<int, int> r(-1, -1);
	if (n <= 0) return r;

	int lower = BSearchLowerBound(A, 0, n - 1, target);
	lower = lower + 1; //move to next element

	if (A[lower] == target)
		r.first = lower;
	else //target is not in the array
		return r;

	int upper = BSearchUpperBound(A, 0, n - 1, target);
	upper = upper < 0 ? (n - 1) : (upper - 1); //move to previous element

	//since in previous search we had check whether the target is
	//in the array or not, we do not need to check it here again
	r.second = upper;

	return r;
}

int SearchInRotatedSortedArray(int array[], int low, int high, int target)
{
	while (low <= high)
	{
		int mid = (low + high) / 2;
		if (target < array[mid])
		if (array[mid] < array[high])//the higher part is sorted
			high = mid - 1; //the target would only be in lower part
		else //the lower part is sorted
		if (target < array[low])//the target is less than all elements in low part
			low = mid + 1;
		else
			high = mid - 1;

		else if (array[mid] < target)
		if (array[low] < array[mid])// the lower part is sorted
			low = mid + 1; //the target would only be in higher part
		else //the higher part is sorted
		if (array[high] < target)//the target is larger than all elements in higher part
			high = mid - 1;
		else
			low = mid + 1;
		else //if(array[mid] == target)
			return mid;
	}

	return -1;
}

int main()
{
	int A[] = { 5, 7, 7, 8, 8, 10 };

	int up=BSearchUpperBound(A,0,5, 8); //5
	int low = BSearchLowerBound(A, 0, 5, 8);//2
	auto range = SearchRange(A, 6, 8); //3,4

	int up1 = BSearchLowerBound(A, 0, 5, 5); //-1
	int low1 = BSearchUpperBound(A, 0, 5, 10); //-1

	// test 非严格上下界
	int up2 = BSearchLowerBound_ext(A, 0, 5, 8); //4
	int low2 = BSearchUpperBound_ext(A, 0, 5, 8);//3

	return 0;
}