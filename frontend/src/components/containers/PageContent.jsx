const PageContent = ({ children, className }) => {
  return (
    <div
    // dark:bg-[#121212] dark:text-slate-50
      className={`p-10 ${className} flex flex-col items-center justify-start w-full`}
    >
      {children}
    </div>
  );
};

export default PageContent;
