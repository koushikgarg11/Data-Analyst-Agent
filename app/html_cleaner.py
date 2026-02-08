"""
HTML preprocessing and cleaning utilities for web scraping tasks.
Simplifies HTML structure to help LLM-generated code find target elements more reliably.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup, Tag
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class HTMLCleaner:
    """Clean and preprocess HTML for data extraction with enhanced Wikipedia support."""
    
    def __init__(self):
        self.removed_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta', 'link', 'noscript']
        self.preserved_tags = ['table', 'tr', 'td', 'th', 'thead', 'tbody', 'caption', 'div', 'span', 'a', 'p']
        
    def clean_html(self, html_content: str, target_keywords: List[str] = None) -> str:
        """
        Clean HTML content for better analysis with enhanced Wikipedia support.
        
        Args:
            html_content: Raw HTML content
            target_keywords: Keywords to help identify relevant sections
            
        Returns:
            Cleaned HTML content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Enhanced Wikipedia-specific cleaning
            if 'wikipedia.org' in html_content.lower():
                self._clean_wikipedia_specific(soup)
            
            # Find and preserve relevant sections
            relevant_sections = self._find_relevant_sections(soup, target_keywords or [])
            
            # If no relevant sections found, try Wikipedia-specific strategies
            if not relevant_sections and 'wikipedia.org' in html_content.lower():
                relevant_sections = self._find_wikipedia_tables(soup, target_keywords or [])
            
            # Create cleaned HTML
            cleaned_soup = self._create_cleaned_html(soup, relevant_sections)
            
            # Simplify table structures
            self._simplify_tables(cleaned_soup)
            
            # Add table finder hints for LLM
            self._add_table_finder_hints(cleaned_soup)
            
            return str(cleaned_soup)
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content
    
    def _clean_wikipedia_specific(self, soup: BeautifulSoup) -> None:
        """Apply Wikipedia-specific cleaning rules."""
        
        # Remove Wikipedia navigation elements
        wiki_nav_selectors = [
            'div.navbox', 'div.infobox', 'div.vertical-navbox',
            'div.navbox-inner', 'table.navbox', 'div.hatnote',
            'div.dablink', 'div.rellink', 'div.noprint',
            'div.printfooter', 'div.catlinks', 'div#mw-navigation'
        ]
        
        for selector in wiki_nav_selectors:
            for elem in soup.select(selector):
                elem.decompose()
        
        # Remove reference sections
        for elem in soup.find_all(['div', 'section'], class_=re.compile(r'references|citations', re.I)):
            elem.decompose()
        
        # Remove edit links
        for elem in soup.find_all('span', class_='mw-editsection'):
            elem.decompose()
        
        # Clean up citation footnotes that clutter tables
        for sup in soup.find_all('sup', class_='reference'):
            sup.decompose()
    
    def _find_wikipedia_tables(self, soup: BeautifulSoup, keywords: List[str]) -> List[Tag]:
        """Find relevant Wikipedia tables using multiple strategies."""
        relevant_tables = []
        all_tables = soup.find_all('table')
        
        # Strategy 1: Look for wikitable class (most data tables have this)
        wikitables = soup.find_all('table', class_=re.compile(r'wikitable', re.I))
        if wikitables:
            relevant_tables.extend(wikitables)
            logger.info(f"Found {len(wikitables)} Wikipedia data tables (wikitable class)")
        
        # Strategy 2: Look for tables with sortable class
        sortable_tables = soup.find_all('table', class_=re.compile(r'sortable', re.I))
        for table in sortable_tables:
            if table not in relevant_tables:
                relevant_tables.append(table)
        
        # Strategy 3: Look for tables with specific content keywords
        for table in all_tables:
            if table in relevant_tables:
                continue
                
            table_text = table.get_text().lower()
            if any(keyword.lower() in table_text for keyword in keywords):
                # Check if table has substantial data (not just formatting)
                rows = table.find_all('tr')
                if len(rows) >= 3:  # At least header + 2 data rows
                    relevant_tables.append(table)
        
        # Strategy 4: Look for tables containing film data patterns
        film_keywords = ['rank', 'title', 'film', 'gross', 'worldwide', 'year', 'peak']
        for table in all_tables:
            if table in relevant_tables:
                continue
                
            # Check headers for film-related terms
            headers = table.find_all('th')
            if headers:
                header_text = ' '.join([th.get_text().lower() for th in headers])
                if sum(1 for keyword in film_keywords if keyword in header_text) >= 3:
                    relevant_tables.append(table)
                    logger.info(f"Found film data table by header analysis")
        
        # Strategy 5: If still no tables, look for largest tables
        if not relevant_tables and all_tables:
            # Sort tables by size (number of cells)
            table_sizes = []
            for table in all_tables:
                cell_count = len(table.find_all(['td', 'th']))
                if cell_count >= 10:  # Only consider tables with meaningful data
                    table_sizes.append((table, cell_count))
            
            # Take the largest tables
            table_sizes.sort(key=lambda x: x[1], reverse=True)
            relevant_tables = [table for table, _ in table_sizes[:3]]  # Top 3 largest
            
            if relevant_tables:
                logger.info(f"Selected {len(relevant_tables)} largest tables as candidates")
        
        return relevant_tables
    
    def _add_table_finder_hints(self, soup: BeautifulSoup) -> None:
        """Add HTML comments to help LLM find tables easily."""
        tables = soup.find_all('table')
        
        for i, table in enumerate(tables):
            # Add a comment before each table describing it
            caption = table.find('caption')
            caption_text = caption.get_text(strip=True) if caption else f"Table {i+1}"
            
            # Count rows and columns
            rows = table.find_all('tr')
            row_count = len(rows)
            col_count = len(rows[0].find_all(['td', 'th'])) if rows else 0
            
            # Get headers
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            header_text = ', '.join(headers[:5]) if headers else 'No headers'
            
            # Create descriptive comment
            comment_text = f"""
DATA_TABLE_{i+1}: {caption_text}
Dimensions: {row_count} rows x {col_count} columns  
Headers: {header_text}
Classes: {' '.join(table.get('class', []))}
"""
            from bs4 import Comment
            comment = Comment(comment_text)
            table.insert_before(comment)
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove scripts, styles, and other unwanted elements."""
        for tag_name in self.removed_tags:
            for element in soup.find_all(tag_name):
                element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
    
    def _find_relevant_sections(self, soup: BeautifulSoup, keywords: List[str]) -> List[Tag]:
        """Find sections that likely contain relevant data."""
        relevant_sections = []
        
        # Strategy 1: Find tables with relevant captions or headers
        tables = soup.find_all('table')
        for table in tables:
            if self._is_relevant_table(table, keywords):
                relevant_sections.append(table)
        
        # Strategy 2: Find divs with relevant content
        divs = soup.find_all('div', class_=re.compile(r'(content|main|article|data|table)', re.I))
        for div in divs:
            if self._contains_relevant_content(div, keywords):
                relevant_sections.append(div)
        
        # If no specific sections found, include all tables
        if not relevant_sections:
            relevant_sections = tables
        
        return relevant_sections
    
    def _is_relevant_table(self, table: Tag, keywords: List[str]) -> bool:
        """Check if a table is likely to contain relevant data."""
        # Check caption
        caption = table.find('caption')
        if caption:
            caption_text = caption.get_text().lower()
            if any(keyword.lower() in caption_text for keyword in keywords):
                return True
        
        # Check table classes
        table_classes = table.get('class', [])
        if any('wikitable' in str(cls).lower() for cls in table_classes):
            return True
        
        # Check headers
        headers = table.find_all(['th'])
        if headers:
            header_text = ' '.join([th.get_text().lower() for th in headers[:5]])
            if any(keyword.lower() in header_text for keyword in keywords):
                return True
        
        # Check if table has substantial data
        rows = table.find_all('tr')
        if len(rows) > 5:  # Tables with multiple rows are likely data tables
            return True
        
        return False
    
    def _contains_relevant_content(self, element: Tag, keywords: List[str]) -> bool:
        """Check if an element contains relevant content."""
        text = element.get_text().lower()
        return any(keyword.lower() in text for keyword in keywords)
    
    def _create_cleaned_html(self, soup: BeautifulSoup, relevant_sections: List[Tag]) -> BeautifulSoup:
        """Create a new HTML document with only relevant sections."""
        # Create new document
        new_soup = BeautifulSoup('<html><head><title>Cleaned Page</title></head><body></body></html>', 'html.parser')
        body = new_soup.find('body')
        
        # Add relevant sections
        for section in relevant_sections:
            # Clone the section to avoid modifying original
            cloned_section = BeautifulSoup(str(section), 'html.parser')
            body.append(cloned_section)
        
        return new_soup
    
    def _simplify_tables(self, soup: BeautifulSoup) -> None:
        """Simplify table structures for easier parsing."""
        tables = soup.find_all('table')
        
        for table in tables:
            # Remove nested tables if they're just formatting
            nested_tables = table.find_all('table')
            for nested in nested_tables:
                if self._is_formatting_table(nested):
                    # Replace with content
                    nested.replace_with(nested.get_text(strip=True))
            
            # Clean up table attributes
            table.attrs = {'class': 'cleaned-table'}
            
            # Simplify cell structures
            for cell in table.find_all(['td', 'th']):
                # Remove styling attributes
                cell.attrs = {}
                
                # Simplify cell content
                self._simplify_cell_content(cell)
    
    def _is_formatting_table(self, table: Tag) -> bool:
        """Check if a table is used only for formatting."""
        rows = table.find_all('tr')
        if len(rows) <= 2:  # Small tables are likely formatting
            return True
        
        # Check if table has data-like content
        cells = table.find_all(['td', 'th'])
        if len(cells) < 4:  # Very few cells suggests formatting table
            return True
        
        return False
    
    def _simplify_cell_content(self, cell: Tag) -> None:
        """Simplify content within a table cell."""
        # Remove sup tags (footnote references)
        for sup in cell.find_all('sup'):
            sup.decompose()
        
        # Simplify links - keep text, remove href
        for link in cell.find_all('a'):
            link.replace_with(link.get_text())
        
        # Remove empty elements
        for elem in cell.find_all():
            if not elem.get_text(strip=True):
                elem.decompose()

def preprocess_html_for_analysis(html_content: str, target_keywords: List[str] = None) -> Tuple[str, Dict]:
    """
    Preprocess HTML content for better analysis results.
    
    Args:
        html_content: Raw HTML content
        target_keywords: Keywords related to the target data
        
    Returns:
        Tuple of (cleaned_html, metadata)
    """
    cleaner = HTMLCleaner()
    
    # Default keywords for common data extraction scenarios
    if not target_keywords:
        target_keywords = [
            'table', 'data', 'list', 'ranking', 'gross', 'revenue', 'sales',
            'statistics', 'results', 'information', 'details', 'overview'
        ]
    
    # Clean the HTML
    cleaned_html = cleaner.clean_html(html_content, target_keywords)
    
    # Extract metadata about the cleaning process
    original_soup = BeautifulSoup(html_content, 'html.parser')
    cleaned_soup = BeautifulSoup(cleaned_html, 'html.parser')
    
    metadata = {
        'original_tables': len(original_soup.find_all('table')),
        'cleaned_tables': len(cleaned_soup.find_all('table')),
        'original_size': len(html_content),
        'cleaned_size': len(cleaned_html),
        'reduction_ratio': 1 - (len(cleaned_html) / len(html_content)) if html_content else 0,
        'target_keywords': target_keywords
    }
    
    logger.info(f"HTML preprocessing complete: {metadata['original_tables']} -> {metadata['cleaned_tables']} tables, "
                f"{metadata['reduction_ratio']:.1%} size reduction")
    
    return cleaned_html, metadata

def extract_table_metadata(html_content: str) -> List[Dict]:
    """
    Extract metadata about tables in HTML content.
    
    Args:
        html_content: HTML content to analyze
        
    Returns:
        List of table metadata dictionaries
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    table_metadata = []
    
    for i, table in enumerate(tables):
        metadata = {
            'index': i,
            'caption': None,
            'headers': [],
            'row_count': 0,
            'col_count': 0,
            'classes': table.get('class', []),
            'has_header_row': False
        }
        
        # Extract caption
        caption = table.find('caption')
        if caption:
            metadata['caption'] = caption.get_text(strip=True)
        
        # Extract headers
        header_cells = table.find_all('th')
        if header_cells:
            metadata['headers'] = [th.get_text(strip=True) for th in header_cells]
            metadata['has_header_row'] = True
        
        # Count rows and columns
        rows = table.find_all('tr')
        metadata['row_count'] = len(rows)
        
        if rows:
            # Estimate column count from first row
            first_row_cells = rows[0].find_all(['td', 'th'])
            metadata['col_count'] = len(first_row_cells)
        
        table_metadata.append(metadata)
    
    return table_metadata

def create_html_summary(html_content: str) -> str:
    """
    Create a summary of HTML structure for LLM analysis.
    
    Args:
        html_content: HTML content to summarize
        
    Returns:
        Text summary of HTML structure
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract page title
    title = soup.find('title')
    title_text = title.get_text() if title else "Unknown"
    
    # Get table metadata
    table_metadata = extract_table_metadata(html_content)
    
    # Create summary
    summary_parts = [
        f"HTML Structure Summary for: {title_text}",
        f"Total Tables Found: {len(table_metadata)}",
        ""
    ]
    
    for i, table_meta in enumerate(table_metadata):
        summary_parts.extend([
            f"Table {i + 1}:",
            f"  Caption: {table_meta['caption'] or 'None'}",
            f"  Headers: {', '.join(table_meta['headers'][:5]) if table_meta['headers'] else 'None'}",
            f"  Dimensions: {table_meta['row_count']} rows × {table_meta['col_count']} columns",
            f"  Classes: {', '.join(table_meta['classes']) if table_meta['classes'] else 'None'}",
            ""
        ])
    
    return "\n".join(summary_parts)

# Integration function for the main scraping workflow
def scrape_and_clean_url(url: str, target_keywords: List[str] = None, save_path: Path = None) -> Tuple[str, Dict, str]:
    """
    Scrape URL and return cleaned HTML with metadata.
    
    Args:
        url: URL to scrape
        target_keywords: Keywords for targeted cleaning
        save_path: Optional path to save cleaned HTML
        
    Returns:
        Tuple of (cleaned_html, metadata, summary)
    """
    try:
        # Scrape the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Clean the HTML
        cleaned_html, metadata = preprocess_html_for_analysis(response.text, target_keywords)
        
        # Create summary
        summary = create_html_summary(cleaned_html)
        
        # Save cleaned HTML if requested
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_html)
            logger.info(f"Cleaned HTML saved to: {save_path}")
        
        return cleaned_html, metadata, summary
        
    except Exception as e:
        logger.error(f"Error scraping and cleaning URL {url}: {e}")
        raise

def create_table_extraction_guide(html_content: str) -> str:
    """
    Create a guide for LLM to help with table extraction from cleaned HTML.
    
    Args:
        html_content: Cleaned HTML content
        
    Returns:
        Text guide for table extraction
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        return "No tables found in the cleaned HTML."
    
    guide_parts = [
        "TABLE EXTRACTION GUIDE:",
        f"Found {len(tables)} table(s) in the cleaned HTML.",
        "",
        "RECOMMENDED EXTRACTION APPROACH:"
    ]
    
    for i, table in enumerate(tables):
        table_info = []
        
        # Get caption
        caption = table.find('caption')
        if caption:
            table_info.append(f"Caption: {caption.get_text(strip=True)}")
        
        # Get headers
        headers = table.find_all('th')
        if headers:
            header_texts = [th.get_text(strip=True) for th in headers[:5]]
            table_info.append(f"Headers: {', '.join(header_texts)}")
        
        # Get dimensions
        rows = table.find_all('tr')
        cols = len(rows[0].find_all(['td', 'th'])) if rows else 0
        table_info.append(f"Size: {len(rows)} rows × {cols} columns")
        
        # Get classes
        classes = table.get('class', [])
        if classes:
            table_info.append(f"CSS Classes: {', '.join(classes)}")
        
        guide_parts.extend([
            f"Table {i+1}:",
            *[f"  {info}" for info in table_info],
            f"  Selector: table:nth-of-type({i+1}) or soup.find_all('table')[{i}]",
            ""
        ])
    
    guide_parts.extend([
        "EXTRACTION TIPS:",
        "1. Use BeautifulSoup to parse: soup = BeautifulSoup(html_content, 'html.parser')",
        "2. Find tables: tables = soup.find_all('table')",
        "3. Look for 'wikitable' class on Wikipedia: soup.find('table', class_='wikitable')",
        "4. Extract headers: headers = [th.get_text(strip=True) for th in table.find_all('th')]",
        "5. Extract rows: rows = [[td.get_text(strip=True) for td in tr.find_all(['td', 'th'])] for tr in table.find_all('tr')]",
        "6. Handle multiple tables by checking captions and headers to find the right one"
    ])
    
    return "\n".join(guide_parts)

def create_wikipedia_film_extraction_guide(html_content: str) -> str:
    """
    Create a specialized extraction guide for Wikipedia highest-grossing films data.
    
    Args:
        html_content: Cleaned HTML content from Wikipedia
        
    Returns:
        Detailed extraction guide for LLM
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    guide_parts = [
        "WIKIPEDIA HIGHEST-GROSSING FILMS EXTRACTION GUIDE:",
        f"Found {len(tables)} table(s) in the cleaned HTML.",
        "",
        "RECOMMENDED EXTRACTION STRATEGY:"
    ]
    
    # Analyze each table for film data characteristics
    for i, table in enumerate(tables):
        table_info = {
            'index': i,
            'caption': None,
            'headers': [],
            'row_count': 0,
            'likely_film_table': False,
            'reasons': []
        }
        
        # Get caption
        caption = table.find('caption')
        if caption:
            caption_text = caption.get_text(strip=True)
            table_info['caption'] = caption_text
            if any(keyword in caption_text.lower() for keyword in ['highest', 'gross', 'film', 'movie', 'box office']):
                table_info['likely_film_table'] = True
                table_info['reasons'].append('Relevant caption')
        
        # Get headers
        headers = table.find_all('th')
        if headers:
            header_texts = [th.get_text(strip=True) for th in headers]
            table_info['headers'] = header_texts
            
            # Check for film-related headers
            header_str = ' '.join(header_texts).lower()
            if any(keyword in header_str for keyword in ['rank', 'film', 'title', 'worldwide', 'gross', 'year', 'peak']):
                table_info['likely_film_table'] = True
                table_info['reasons'].append('Film-related headers')
        
        # Get dimensions
        rows = table.find_all('tr')
        table_info['row_count'] = len(rows)
        
        # Check for substantial data (film tables usually have many rows)
        if len(rows) > 20:
            table_info['likely_film_table'] = True
            table_info['reasons'].append(f'Large dataset ({len(rows)} rows)')
        
        # Check for classes
        classes = table.get('class', [])
        if any('wikitable' in str(cls).lower() for cls in classes):
            table_info['likely_film_table'] = True
            table_info['reasons'].append('Wikipedia data table class')
        
        # Add table info to guide
        guide_parts.extend([
            f"Table {i+1}:",
            f"  Caption: {table_info['caption'] or 'None'}",
            f"  Headers: {', '.join(table_info['headers'][:6]) if table_info['headers'] else 'None'}",
            f"  Rows: {table_info['row_count']}",
            f"  CSS Classes: {', '.join(classes) if classes else 'None'}",
            f"  Likely Film Data: {'YES' if table_info['likely_film_table'] else 'NO'}",
            f"  Reasons: {', '.join(table_info['reasons']) if table_info['reasons'] else 'None'}",
            ""
        ])
    
    # Find the most likely table
    best_table_index = None
    best_score = 0
    
    for i, table in enumerate(tables):
        score = 0
        
        # Score based on various factors
        caption = table.find('caption')
        if caption and any(keyword in caption.get_text().lower() for keyword in ['highest', 'gross', 'film']):
            score += 10
        
        headers = table.find_all('th')
        if headers:
            header_text = ' '.join([th.get_text().lower() for th in headers])
            if 'rank' in header_text: score += 5
            if 'film' in header_text or 'title' in header_text: score += 5
            if 'gross' in header_text or 'worldwide' in header_text: score += 5
            if 'year' in header_text: score += 3
        
        rows = table.find_all('tr')
        if len(rows) > 50: score += 5  # Large datasets are likely the main table
        elif len(rows) > 20: score += 3
        
        classes = table.get('class', [])
        if any('wikitable' in str(cls).lower() for cls in classes):
            score += 3
            
        if score > best_score:
            best_score = score
            best_table_index = i
    
    guide_parts.extend([
        "RECOMMENDED TABLE:",
        f"Based on analysis, Table {best_table_index + 1} appears to be the main film data table." if best_table_index is not None else "No clear film data table identified.",
        "",
        "EXTRACTION CODE TEMPLATE:",
        "```python",
        "# Find the target table",
        f"tables = soup.find_all('table')",
        f"target_table = tables[{best_table_index}] if len(tables) > {best_table_index} else None" if best_table_index is not None else "target_table = None",
        "",
        "# Alternative: Find by characteristics",
        "for table in tables:",
        "    caption = table.find('caption')",
        "    if caption and 'highest' in caption.get_text().lower():",
        "        target_table = table",
        "        break",
        "    ",
        "    headers = [th.get_text().strip() for th in table.find_all('th')]",
        "    if any('rank' in h.lower() for h in headers) and any('gross' in h.lower() for h in headers):",
        "        target_table = table",
        "        break",
        "",
        "# Extract data",
        "if target_table:",
        "    df = pd.read_html(str(target_table))[0]",
        "    # Clean and process the data...",
        "```",
        "",
        "COMMON ISSUES & SOLUTIONS:",
        "1. If pd.read_html fails: Extract manually using BeautifulSoup",
        "2. If table has merged cells: Look for rowspan/colspan attributes",
        "3. If headers are complex: Clean header text and create simple column names",
        "4. If data has footnotes: Remove <sup> tags before extraction",
        "5. If no clear table found: Try looking for div elements with table-like structure"
    ])
    
    return "\n".join(guide_parts)
