const PageContent = ({ children, className }) => {
  return (
    <div
      className={`p-10 ${className} flex flex-col items-center justify-start w-full`}
    >
      {children}
    </div>
  );
};

export default PageContent;
